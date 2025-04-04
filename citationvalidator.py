import hashlib
import os
import statistics
import sys
import time
from pprint import pprint
import crossref_commons.retrieval
import re
from tabulate import tabulate
from tqdm import tqdm as original_tqdm
from dynamic_multiprocessing import dynamic_multiprocessing
from urllib.request import urlopen
import urllib.error
from urllib.parse import quote, urlencode
import pandas as pd
import json
import csv
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
# from scholarly import scholarly
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

def tqdm(*args, **kwargs):
	kwargs.setdefault("ncols", 100)
	kwargs.setdefault("file", sys.stdout)
	return original_tqdm(*args, **kwargs)

# PMC_service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=windows&email=frederik.bay2@gmail.com"


def load_file(file: str):
	print("Reading file:", file)
	with open(file, "r", encoding="utf-8") as f:
		txt = f.read()
	return txt


def load_bibtex(file: str, debug = None):
	# txt_divided is a list containing the full text divided into the bibliographic entries
	# bib_types is a list containing tuples with the type and name respectively of the bib entry
	# bibs is a dict with the bib name containing a dict with that bib's keys and values
	txt = load_file(file)
	
	# Make sure that there's a newline at the end, and remove all comments outside of bib entries
	if not txt.endswith("\n"):
		if debug: print("Adding extra newline at the end")
		txt += "\n"
	txt = re.sub(r"(?<=\})([^@=]*?)(?=@)", "\n\n", txt, flags=re.S)
	
	txt_divided = {bib_name: bib_entry for bib_entry, bib_name in re.findall(r"(@\w+\{(.+?),.+?(?:\}[\s\n]*)+\n)(?=[\s\n]*(?:@|$))", txt, flags=re.S)}
	if debug: print(txt_divided[debug])
	bib_types = {bib_name: bib_type for bib_type, bib_name in [re.findall(r"@(\w+)\{(.+?),", bib_entry)[0] for bib_name, bib_entry in txt_divided.items()]}
	if debug: print(bib_types[debug])
	# bibs = {bib_name : {key.lower() : value for key, value in re.findall(r"(\w+)\s?=\s?[\"\{]*([^=\{\}\"]*)[\}\"]*,", txt_divided[item_i])} for item_i, bib_name in enumerate(bib_types)}
	if debug: print({bib_name: {key.lower(): value for key, value in re.findall(r"(\w+)\s*=\s*([\s\S]*?)(?=\s*,(?=[\s\n]*\w+\s*=)|\}\n)", txt_divided[bib_name])} for bib_name in bib_types}[debug])  # Print without clean_text()
	bibs = {bib_name: {key.lower(): clean_text(value) for key, value in re.findall(r"(\w+)\s*=\s*([\s\S]*?)(?=\s*,(?=[\s\n]*\w+\s*=)|\}\n)", txt_divided[bib_name])} for bib_name in bib_types}
	if debug: print(bibs[debug])
	return bibs, bib_types

def latex2citations_statements(txt):
	rough_division = re.split(r"\\cite(?:p|t)?\{(.*?)\}\.?", txt, flags=re.S)  # Text segments: even indices; citations: odd indices.
	text_segments, citation_segments = rough_division[::2], rough_division[1::2]
	# Cropping text_segments and citation_segments
	text_segments_cropped = [re.findall(r".*(?:(?:\n[\t ]*){2}|\\\\|^)(.*?)$", text_segment, flags=re.S)[0] for text_segment in text_segments]
	text_segments_cropped = [re.sub(r"\\(?!text(?:bf|it)?)\w+[\{\[](.*?)[\}\]]", "", text_segment, flags=re.S) for text_segment in text_segments_cropped]  # Remove section headers, labels etc.
	text_segments_cropped = [re.sub(r"\\text(?:bf|it)?\{(.*?)\}", r"\1", text_segment, flags=re.S) for text_segment in text_segments_cropped]  # Replace bold, italic (etc.) text with just the text.
	text_segments_cropped = [clean_text(text_segment_cropped) for text_segment_cropped in text_segments_cropped]
	citation_segments_cropped = [re.findall(r"\s*([^,]+)", citation_segment) for citation_segment in citation_segments]
	citations, statements = [], []
	for citation_segment, text_segment in zip(citation_segments_cropped, text_segments_cropped):
		if len(citation_segment) > 1:
			for citation_subsegment in citation_segment:
				citations.append(citation_subsegment)
				statements.append(text_segment)
		else:
			citations.append(citation_segment[0])
			statements.append(text_segment)
	
	return citations, statements


def save_bibtex(bib_dict, bib_types, filename):
	# Convert dictionaries to BibTeX string
	print("Converting dict to BibTeX string...")
	bibtex_string = dict_to_bibtex(bib_dict, bib_types)
	
	# Write BibTeX string to a file
	print("Saving new BibTeX file:", filename)
	with open(filename, "w", encoding="utf8") as bibfile:
		bibfile.write(bibtex_string)
	
	print("BibTeX file created successfully.")


def hash_url(url):
	return hashlib.md5(url.encode()).hexdigest()

def get_html_from_url(url, retrieve_from_cache=True, save_to_cache=True):
	html = None
	headers = None
	url = url.replace(" ", "%20").replace("‐", "-")  # TODO THIS! in citations
	
	cache_folder = "cached_urls"
	html_filename = hash_url(url) + "_html.txt"
	headers_filename = hash_url(url) + "_headers.json"
	html_filepath = os.path.join(cache_folder, html_filename)
	headers_filepath = os.path.join(cache_folder, headers_filename)
	
	# Retrieving cached html and header
	if retrieve_from_cache:
		# Check if the files already exist
		if os.path.exists(html_filepath) and os.path.exists(headers_filepath):
			# print(f"Loading content from {html_filepath} and {headers_filepath}")
			with open(html_filepath, 'r', encoding='utf-8') as html_file:
				html = html_file.read()
			with open(headers_filepath, 'r', encoding='utf-8') as headers_file:
				headers = json.load(headers_file)
			# return html, headers
	
	# Fetch from url if retrieve_from_cache didn't work
	if not html:
		print("Fetching from URL")
		# try:
		# Fetching HTML and headers
		response = urlopen(url)
		html = response.read().decode('utf-8')
		headers = dict(response.getheaders())
		# except Exception as e:
		# 	print(f"Failed to fetch {url}: {e}")
		# 	return None, None
		
		if save_to_cache:
			# Create the cache directory if it doesn't exist
			if not os.path.exists(cache_folder):
				os.makedirs(cache_folder)
			
			# Save HTML content
			with open(html_filepath, 'w', encoding='utf-8') as html_file:
				html_file.write(html)
			
			# Save headers
			with open(headers_filepath, 'w', encoding='utf-8') as headers_file:
				json.dump(headers, headers_file)
			
			# print(f"Saved {url} content to {html_filepath} and {headers_filepath}")
		# return html, headers
	
	# page = urlopen(url)
	# headers = dict(page.getheaders())
	# html_bytes = page.read()
	# html = html_bytes.decode("utf-8")
	return html, headers

def get_DOI_by_title_from_SciHub(title):
	url = "https://sci-hub.se/"

crossref_response_times = []
def get_DOI_by_title_author_from_crossref(title = "", author = None):
	DOI = None
	url = "https://api.crossref.org/works?mailto=frederik.bay2@gmail.com&rows=1" + bool(title) * f"&query.title={title}" + bool(author) * f"&query.author={author}"
	
	# If response times are getting longer, sleep a bit
	# if len(crossref_response_times) > 0 and crossref_response_times[-1] > 10 and crossref_response_times[-1] > 1.10 * statistics.median(crossref_response_times):
	# 	print(f"Crossref response time is going up (current: {round(crossref_response_times[-1], 2)}s, median: {round(statistics.median(crossref_response_times), 2)}s)")
	# 	# time.sleep(statistics.median(crossref_response_times))
	# 	# time.sleep(2)
	try:
		time_0 = time.perf_counter()
		html, headers = get_html_from_url(url)
		if time.perf_counter() - time_0 > 0.2: crossref_response_times.append(time.perf_counter() - time_0)
	except urllib.error.HTTPError as e:
		if e.code == 504:
			print("504 Gateway Time-out error. Retrying...")
			time.sleep(5)
			try:
				time_0 = time.perf_counter()
				html, headers = get_html_from_url(url)
				if time.perf_counter() - time_0 > 0.2: crossref_response_times.append(time.perf_counter() - time_0)
			except urllib.error.HTTPError as e:
				if e.code == 504:
					print("504 again. Retrying one last time...")
					time.sleep(30)
					time_0 = time.perf_counter()
					html, headers = get_html_from_url(url)
					if time.perf_counter() - time_0 > 0.2: crossref_response_times.append(time.perf_counter() - time_0)
				else:
					raise
				
		else:
			raise
	
	if headers["x-api-pool"] != "polite":
		print("Not in polite API pool.")
	
	result = json.loads(html)
	num_results = result["message"]["total-results"]
	if num_results > 0:
		result_title = result["message"]["items"][0]["title"][0]
		if get_simple_overlap_score(title, result_title) >= 0.90:
			# Titles are assumed to be the same
			DOI = result["message"]["items"][0]["DOI"]
	
	return DOI

def is_valid_DOI_format(DOI):
	return re.match(r"^10.\d{4,9}/[-._;()/:A-Z0-9]+$", DOI, flags=re.I)

def get_PMID_from_DOI(DOI):
	# Pre-check that it is a valid DOI
	if not is_valid_DOI_format(DOI):
		raise ValueError(f"DOI ({DOI}) is not a valid format")  # Maybe raise an error instead? (since it actually basically is an error)
	PMID = None
	url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=" + DOI + "&format=xml"
	html, headers = get_html_from_url(url)
	if not ("No items found" in html or "invalid article id" in html):
		result = re.findall(r"\<Id\>(\d+)\<\/Id\>", html)
		if len(result) == 0:
			raise ValueError("No PMID returned")
		elif len(result) > 1:
			raise ValueError("Too many PMIDs returned")
		else:
			PMID = result[0]
	return PMID

def get_PMID_by_title(title):
	PMID = None
	url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?tool=windows&email=frederik.bay2@gmail.com&retmax=1&term="' + title + '"[Title:~0]'
	html, headers = get_html_from_url(url)
	# If more than 1 result, don't return any
	if int(next(iter(re.findall(r"\<count\>(\d+)\<\/count\>", html, flags=re.I)), 0)) > 1:
		raise IndexError("Too many results returned")
		# return None
	PMID = next(iter(re.findall(r"\<Id\>(\d+)\<\/Id\>", html)), None)
	return PMID

def get_abstract_by_PMID(PMID):
	abstract = None
	url = "https://pubmed.ncbi.nlm.nih.gov/" + PMID
	html, headers = get_html_from_url(url)
	abstract = next(iter(re.findall(r"\<div.*?class=\"abstract\".*?\<(?:p|em).*?\>[\s\n]*(.*)[\s\n]*\<\/(?:p|em).*?\>[\s\n]*\<\/div", html, flags=re.S | re.I)), None)
	return abstract

def get_abstract_by_DOI(DOI):
	abstract = None
	url = "http://dx.doi.org/" + DOI
	html, headers = get_html_from_url(url)
	abstract = next(iter(re.findall(r"abstract.*?\<p.*?\>(.*)\<\/p", html, flags=re.I)), None)
	return abstract


def get_DOIs(bibs, allow_copying_existing=False):
	DOIs = {}
	pbar = tqdm(total=len(bibs))
	for bib_name, bib_entry in bibs.items():
		pbar.set_description(bib_name)
		DOI = bib_entry.get("doi", "") if allow_copying_existing else ""
		title = bib_entry.get("title", "")
		
		DOI_result = ""
		# Get DOI from Crossref
		if title and (not DOI or DOI and not is_valid_DOI_format(DOI)):
			try:
				DOI_result = get_DOI_by_title_author_from_crossref(title, author=bib_entry.get("author", None))
				if not DOI_result and "author" in bib_entry.keys():
					# Try again without the author (making sure that the author field was in fact there before)
					DOI_result = get_DOI_by_title_author_from_crossref(title)
					if not DOI_result:
						print(f"DOI could not be found from title.")
			except Exception as e:
				print(f"DOI could not be found from title: {e}.")
		DOIs[bib_name] = DOI_result if DOI_result else DOI
		pbar.update()
	pbar.set_description("DOIs")
	pbar.close()
	return DOIs

def get_PMIDs(bibs, allow_copying_existing=False):
	PMIDs = {}
	pbar = tqdm(total=len(bibs))
	for bib_name, bib_entry in bibs.items():
		pbar.set_description(bib_name)
		PMID = bib_entry.get("pmid", "") if allow_copying_existing else ""
		DOI = bib_entry.get("doi", "")
		title = bib_entry.get("title", "")
		
		PMID_result = ""
		# Get PMID from DOI
		if DOI and not PMID:
			try:
				PMID_result = get_PMID_from_DOI(DOI)
				if not PMID_result:
					print(f"PMID could not be found from DOI.")
			except Exception as e:
				print(f"PMID could not be found from DOI: {e}.")
		
		# Get PMID from title if PMID is still not found
		if title and not PMID_result and not PMID:
			try:
				PMID_result = get_PMID_by_title(title)
				if not PMID_result:
					print(f"PMID could not be found from title.")
			except Exception as e:
				print(f"PMID could not be found from title: {e}.")
		
		PMIDs[bib_name] = PMID_result if PMID_result else PMID
		pbar.update()
	pbar.set_description("PMIDs")
	pbar.close()
	return PMIDs

def get_reference_and_citation_counts(bibs, allow_copying_existing=False):
	reference_counts = {}
	citation_counts = {}
	pbar = tqdm(total=len(bibs))
	for bib_name, bib_entry in bibs.items():
		pbar.set_description(bib_name)
		PMID = bib_entry.get("pmid", "")
		DOI = bib_entry.get("doi", "")
		title = bib_entry.get("title", "")
		reference_count = bib_entry.get("reference_count", "") if allow_copying_existing else ""
		citation_count = bib_entry.get("citation_count", "") if allow_copying_existing else ""
		
		reference_count_result, citation_count_result = "", ""
		
		# Get reference count from Crossref and citation count from Opencitations
		# Get reference count from Crossref
		if DOI and not reference_count:
			try:
				url = "https://api.crossref.org/works/" + DOI + "?mailto=frederik.bay2@gmail.com"
				html, headers = get_html_from_url(url)
				result = json.loads(html)
				reference_count_result = str(result["message"]["reference-count"])
				if not reference_count_result:
					print(f"Reference count could not be found.")
			except Exception as e:
				print(f"Reference count could not be found: {e}.")
			
		if not reference_count and not reference_count_result:  # If still no result, use the Python API
			print("Using Python Crossref API instead of URL")
			try:
				crossref_result = crossref_commons.retrieval.get_publication_as_json(DOI)
				reference_count_result = str(crossref_result["reference-count"])
				if not reference_count_result:
					print(f"Reference count could not be found.")
			except Exception as e:
				print(f"Reference count could not be found: {e}.")
		
		# # Get citation count from Google Scholar/Scholarly
		# if False:
		# 	try:
		# 		s = next(scholarly.search_pubs(bib["doi"]))
		# 		citation_count = str(s["num_citations"])
		# 	except Exception as e:
		# 		print("\n" * no_prints_yet + "Citation count could not be found:", e);  no_prints_yet = False
		
		# Get citation count from Opencitations
		if DOI and not citation_count:
			try:
				url = "https://opencitations.net/index/coci/api/v1/citations/" + DOI
				html, headers = get_html_from_url(url)
				citation_result = json.loads(html)
				citation_count_result = str(len(citation_result))
				if not citation_count_result:
					print(f"Citation count could not be found.")
			except Exception as e:
				print(f"Citation count could not be found: {e}.")
		
		reference_counts[bib_name] = reference_count_result if reference_count_result else reference_count
		citation_counts[bib_name] = citation_count_result if citation_count_result else citation_count
		pbar.update()
	pbar.set_description("Ref & cite counts")
	pbar.close()
	return reference_counts, citation_counts


def get_abstracts(bibs, allow_copying_existing=True):
	abstracts = {}
	fails = {}
	pbar = tqdm(total=len(bibs))
	for bib_name, bib_entry in bibs.items():
		pbar.set_description(bib_name)
		abstract = bib_entry.get("abstract", "") if allow_copying_existing else ""
		PMID = bib_entry.get("pmid", "")
		DOI = bib_entry.get("doi", "")
		title = bib_entry.get("title", "")
		
		no_prints_yet = True
		
		abstract_result = ""
		abstract_status = False  # bool(abstract)
		
		# If the abstract is already in the bib, copy it
		if not abstract_status and allow_copying_existing and abstract:
			abstract_result = abstract
			abstract_status = True
		
		# If PMID: Get abstract from PMID
		if not abstract_status and PMID:
			abstract_result = get_abstract_by_PMID(PMID)
			if not abstract_result:
				abstract += "Abstract via PMID failed. "
				print("\n" * no_prints_yet + "PMID failed.");  no_prints_yet = False
			elif "No abstract available" in abstract_result:
				abstract += "PMID: 'No abstract available'. "
				print("\n" * no_prints_yet + "PMID: 'No abstract available'.");  no_prints_yet = False
			else:
				abstract = abstract_result
				abstract_status = True
		# else:
		# 	abstract += "No PMID. "
			
		# Try scraping DOI landing page for abstract
		if not abstract_status and DOI:
			print("\n" * no_prints_yet + "Trying DOI.", end="");  no_prints_yet = False
			try:
				abstract_result = get_abstract_by_DOI(DOI)
				if not abstract_result:
					abstract += "DOI failed. "
					print(f"\rDOI failed.")
				else:
					abstract = abstract_result
					abstract_status = True
					print(" Successful.")
			except Exception as e:
				abstract += f"DOI failed: {e}. "
				print(f"\rDOI failed:", e)
		# else:
		# 	abstract += "No DOI. "
		
		if not abstract_status:  # Add "ERROR" in the beginning if abstract couldn't be found
			abstract = "ERROR: " + abstract
		
		abstract = clean_text(abstract)
		
		abstracts[bib_name] = abstract
		if not abstract_status:
			fails[bib_name] = abstract
		
		pbar.update()
	pbar.set_description("Abstracts")
	pbar.close()
	
	if fails:
		print(f"Some abstracts couldn't be found:  {len(fails)} / {len(bibs)} ({round(len(fails) / len(bibs) * 100, 1)}%)")
		print(tabulate([[bib_name, err_msg] for bib_name, err_msg in fails.items()], headers=["Bib name", "Error message"]))
	# print(f"Number of failed abstracts: {len(fails)} / {len(bibs)} ({round(len(fails) / len(bibs) * 100, 1)}%)")
	
	return abstracts


def go(bib, allow_copying_existing_abstract=True):
	# crossref_result = crossref_commons.retrieval.get_publication_as_json("10.1038/cr.2007.113")
	# abstracts.append(crossref_result)
	
	no_prints_yet = True
	DOI = bib.get("doi", "").strip()
	PMID = bib.get("pmid", "").strip()
	abstract = ""
	reference_count = ""
	citation_count = ""
	abstract_status = False
	abstract_status_msg = ""  # This and/or keep it in abstract?
	
	
	# If not DOI: Get DOI from crossref
	if ("doi" in bib.keys() and bib["doi"].strip() == "" or "doi" not in bib.keys()) and "title" in bib.keys() and bib["title"].strip() != "" or not is_valid_DOI_format(DOI):
		try:
			DOI_result = get_DOI_by_title_author_from_crossref(bib["title"], author=bib.get("author", None))
			if not DOI_result and "author" in bib.keys():
				# Try again without the author (making sure that the author field was in fact there before)
				DOI_result = get_DOI_by_title_author_from_crossref(bib["title"])
				if not DOI_result:
					abstract += "DOI could not be found from title. "
					print("\n" * no_prints_yet + "DOI could not be found from title.");  no_prints_yet = False
			DOI = DOI_result if DOI_result else DOI
		except Exception as e:
			abstract += f"DOI could not be found from title: {e}. "
			print("\n" * no_prints_yet + "DOI could not be found from title:", e);  no_prints_yet = False
	
	
	# If DOI and not PMID: Get PMID from DOI
	if DOI or "doi" in bib.keys() and bib["doi"].strip() != "" and ("pmid" not in bib.keys() or "pmid" in bib.keys() and bib["pmid"].strip() == ""):
		try:
			PMID = get_PMID_from_DOI(DOI)
			if not PMID:
				abstract += "PMID could not be found from DOI. "
				print("\n" * no_prints_yet + "PMID could not be found from DOI.");  no_prints_yet = False
		except Exception as e:
			abstract += f"PMID could not be found from DOI: {e}. "
			print("\n" * no_prints_yet + "PMID could not be found from DOI:", e);  no_prints_yet = False
	
	
	# From here trying to fetch abstract
	# If the abstract is already in the bib, copy it
	if not abstract_status and "abstract" in bib.keys() and bib["abstract"].strip() != "" and allow_copying_existing_abstract:
		# print("\n" * no_prints_yet + "Copying existing abstract.", end="");  no_prints_yet = False
		abstract = bib["abstract"]
		abstract_status = True  # TODO: Is it really? Everything else failed, and now it just copied whatever was already there. Also it should be renamed to abstract_status
		# print(" Successful.")
	
	# If PMID: Get abstract from PMID
	if not abstract_status and ("pmid" in bib.keys() and bib["pmid"] != "" or PMID):
		abstract_result = get_abstract_by_PMID(PMID)
		if not abstract_result:
			abstract += "Abstract via PMID failed. "
			print("\n" * no_prints_yet + "PMID failed.");  no_prints_yet = False
		elif "No abstract available" in abstract_result:
			abstract += "PMID: No abstract available. "
			print("\n" * no_prints_yet + "PMID: No abstract available.");  no_prints_yet = False
		else:
			abstract = abstract_result
			abstract_status = True
	
	# Try scraping DOI landing page for abstract
	if not abstract_status and ("doi" in bib.keys() and bib["doi"].strip() != "" or DOI):
		print("\n" * no_prints_yet + "Trying DOI.", end="");  no_prints_yet = False
		try:
			abstract_result = get_abstract_by_DOI(DOI)
			if not abstract_result:
				abstract += "DOI failed. "
				print(f"\rDOI failed.")
			else:
				abstract = abstract_result
				abstract_status = True
				print(" Successful.")
		except Exception as e:
			abstract += f"DOI failed: {e}. "
			print(f"\rDOI failed:", e)
	
	# Try searching for title on PubMed
	if not abstract_status:
		print("\n" * no_prints_yet + "Searching for title on PubMed.", end="");  no_prints_yet = False
		try:
			PMID = get_PMID_by_title(bib["title"])
			if PMID:
				try:
					abstract_result = get_abstract_by_PMID(PMID)
					if not abstract_result:
						abstract += "Abstract via PMID via PubMed title search failed. "
						print("\rPubMed title search failed.")
					elif "No abstract available" in abstract_result:
						abstract += "PMID title search: No abstract available. "
						print("\rPMID title search: No abstract available.");  no_prints_yet = False
					else:
						abstract = abstract_result
						abstract_status = True
						print(" Successful.")
				except Exception as e:
					abstract += f"Abstract via PMID via PubMed title search failed: {e}. "
					print("\rAbstract via PMID via PubMed title search failed:", e)
			else:
				abstract += "PMID could not be found from title. "
				print("\rPMID could not be found from title.")
		except Exception as e:
			abstract += f"PMID could not be found from title: {e}. "
			print("\rPMID could not be found from title:", e)
	
	
	if not abstract_status:  # Add "ERROR" in the beginning if abstract couldn't be found
		abstract = "ERROR: " + abstract
	
	
	# Get reference count and citation count
	if "doi" in bib.keys() and bib["doi"].strip() != "":
		# Get reference count from Crossref
		if "reference_count" in bib.keys() and bib["reference_count"].strip() == "" or "reference_count" not in bib.keys():
			try:
				crossref_result = crossref_commons.retrieval.get_publication_as_json(bib["doi"])
				reference_count = str(crossref_result["reference-count"])
			except Exception as e:
				print("\n" * no_prints_yet + "Reference count could not be found:", e);  no_prints_yet = False
		
		# Get citation count from Google Scholar/Scholarly
		# if False:
		# 	try:
		# 		s = next(scholarly.search_pubs(bib["doi"]))
		# 		citation_count = str(s["num_citations"])
		# 	except Exception as e:
		# 		print("\n" * no_prints_yet + "Citation count could not be found:", e);  no_prints_yet = False
		
		# Get citation count from Opencitations
		if "citation_count" in bib.keys() and bib["citation_count"].strip() == "" or "citation_count" not in bib.keys():
			try:
				url = "https://opencitations.net/index/coci/api/v1/citations/" + bib["doi"]
				html, headers = get_html_from_url(url)
				citation_result = json.loads(html)
				citation_count = str(len(citation_result))
			except Exception as e:
				print("\n" * no_prints_yet + "Citation count could not be found:", e);  no_prints_yet = False
	
	
	return abstract, DOI, PMID, abstract_status, reference_count, citation_count

def run_go(bibs, allow_copying_existing_abstract=True, multiprocessing=False):
	PMIDs = {}
	DOIs = {}
	PMCIDs = {}
	abstracts = {}
	reference_counts = {}
	citation_counts = {}
	fails = {}
	bib_names = list(bibs)
	print("Fetching abstracts")
	if multiprocessing:
		for i_bib, go_result in enumerate(dynamic_multiprocessing(bibs.values(), go, True, max_processes=3, tqdm_desc="Bibs")):
			bib_name = bib_names[i_bib]
			abstracts[bib_name], DOIs[bib_name], PMIDs[bib_name], abstract_status, reference_counts[bib_name], citation_counts[bib_name] = go_result
	else:
		pbar = tqdm(total=len(bibs), ncols=100, file=sys.stdout)
		for i_bib, [bib_name, bib_entry] in enumerate(bibs.items()):
			pbar.set_description(bib_name)
			abstracts[bib_name], DOIs[bib_name], PMIDs[bib_name], abstract_status, reference_counts[bib_name], citation_counts[bib_name] \
				= go(bib_entry, allow_copying_existing_abstract=allow_copying_existing_abstract)
			if not PMIDs[bib_name]: del PMIDs[bib_name]
			if not abstract_status: fails[bib_name] = abstracts[bib_name]
			pbar.update()
		
		pbar.close()
	
	print("Number of failed entries:", len(fails))
	if fails:
		for key, value in fails.items():
			print(f"  {key} : {value}")
	
	return abstracts, DOIs, PMIDs, reference_counts, citation_counts


def find_matching_bibs(bibs):
	# Check exact matches between DOIs
	DOIs = {bib_name: bib_entry["doi"] for bib_name, bib_entry in bibs.items() if "doi" in bib_entry.keys()}
	matching_DOIs = {}
	for DOI in DOIs.values():
		if sum([DOI == x for x in list(DOIs.values())]) > 1:  # If there are more than two identical DOIs in the list
			matching_DOIs[DOI] = [bib_name for bib_name, bib_value in DOIs.items() if bib_value == DOI]
	
	print("Exact matching DOIs:")
	for DOI, bib_names in matching_DOIs.items():
		print(f"{DOI} : {", ".join(bib_names)}")
	
	# Compare titles (other properties)
	...


def TF_IDF_match_score_statement_vs_abstract(statement, abstract):
	# Get a match score between the statement in a LaTeX file and the abstract of the following citation
	# From https://stackoverflow.com/a/8897648
	corpus = [statement, abstract]
	vect = TfidfVectorizer(min_df=1)
	tfidf = vect.fit_transform(corpus)
	pairwise_similarity = tfidf * tfidf.T
	# Only need these next lines if comparing multiple texts to find the most similar one
	# n, _ = pairwise_similarity.shape
	# pairwise_similarity[np.arange(n), np.arange(n)] = -1.0
	# pairwise_similarity[input_idx].argmax()
	return pairwise_similarity[0,1]


def citation_abstract_score_matching(statements: list, abstracts: dict):
	used_bib_names = [bib_name for bib_name, _ in statements if bib_name in abstracts]
	
	scores = []
	collected_list = []
	for bib_name, statement in tqdm(statements, ncols=100, file=sys.stdout):
		if bib_name in abstracts:
			
			TF_IDF_score = TF_IDF_match_score_statement_vs_abstract(statement, abstracts[bib_name])
		else:  # no abstract
			TF_IDF_score = -1
			abstracts[bib_name] = ""  # Just to not get an error in collected_list
		scores.append(TF_IDF_score)
		collected_list.append([TF_IDF_score, bib_name, statement, abstracts[bib_name]])
	
	# collected_list = [[score, bib_name, statement, abstract] for score, [bib_name, statement], abstract in zip(scores, statements, abstracts)]
	return collected_list

def get_TF_IDF_scores(citations_and_statements: list, abstracts: dict):
	scores = []
	for bib_name, statement in tqdm(citations_and_statements, desc="TF_IDF", ncols=100, file=sys.stdout):
		if bib_name not in abstracts:
			abstracts[bib_name] = ""
		score = TF_IDF_match_score_statement_vs_abstract(statement, abstracts[bib_name])
		scores.append(score)
	
	return scores


def get_BERT_scores(citations_and_statements: list, abstracts: dict):
	model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
	scores = []
	for bib_name, statement in tqdm(citations_and_statements, desc="BERT", ncols=100, file=sys.stdout):
		if bib_name not in abstracts:
			abstracts[bib_name] = ""
		embeddings = model.encode([statement, abstracts[bib_name]])
		score = util.cos_sim(embeddings[0], embeddings[1]).item()
		scores.append(score)
	
	return scores


def get_BioBERT_scores(citations_and_statements: list, abstracts: dict):
	model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
	scores = []
	for bib_name, statement in tqdm(citations_and_statements, desc="BioBERT", ncols=100, file=sys.stdout):
		if bib_name not in abstracts:
			abstracts[bib_name] = ""
		embeddings = model.encode([statement, abstracts[bib_name]])
		score = util.cos_sim(embeddings[0], embeddings[1]).item()
		scores.append(score)
	
	return scores


def get_simple_overlap_score(text1, text2):
	text1_words = re.findall(r"\w+", text1.lower(), flags=re.I)
	text2_words = re.findall(r"\w+", text2.lower(), flags=re.I)
	text1_words_set = set(text1_words)
	text2_words_set = set(text2_words)
	common_words = text1_words_set.intersection(text2_words_set)
	text1_num_words = len(text1_words)
	text2_num_words = len(text2_words)
	score = len(common_words) / len(text1_words_set)
	return score

def get_simple_overlap_scores(citations_and_statements: list, abstracts: dict):
	scores = []
	for bib_name, statement in tqdm(citations_and_statements, desc="Overlap", ncols=100, file=sys.stdout):
		if bib_name not in abstracts:
			abstracts[bib_name] = ""
		statement_words = re.findall(r"\w+", statement.lower(), flags=re.I)
		abstract_words = re.findall(r"\w+", abstracts[bib_name].lower(), flags=re.I)
		statement_words_set = set(statement_words)
		abstract_words_set = set(abstract_words)
		common_words = statement_words_set.intersection(abstract_words_set)
		statement_num_words = len(statement_words)
		abstract_num_words = len(abstract_words)
		score = len(common_words) / len(statement_words_set)
		scores.append(score)
	
	return scores


def get_fuzzy_score(str1, str2):
	return SequenceMatcher(None, str1, str2).ratio()


# Create .bib file with abstracts
def dict_to_bibtex(bib_dict, bib_types):
	bibtex_entries = []
	for key, entry in bib_dict.items():
		entry_type = bib_types.get(key, "misc")
		bibtex_entry = f"@{entry_type}{{{key},\n"
		for field, value in entry.items():
			if field == "title":
				bibtex_entry += f"  {field} = \"{{{value}}}\",\n"
			else:
				bibtex_entry += f"  {field} = {{{value}}},\n"
		bibtex_entry = bibtex_entry.rstrip(',\n') + "\n}\n\n"
		bibtex_entries.append(bibtex_entry)
	bibtex_string = "\n".join(bibtex_entries).strip() + "\n"
	bibtex_string = bibtex_string.replace(r"%", r"\%")
	return bibtex_string

# Add a new property (e.g. "abstract") to the bibs
def add_prop_to_bib_entries(bibs: dict, property_key: str, property_dict: dict, replace_existing=False):
	for bib_name, property_value in property_dict.items():
		if bib_name in bibs and (property_key not in bibs[bib_name].keys() or bibs[bib_name][property_key].strip() == "" or replace_existing):
			bibs[bib_name][property_key] = property_value
	return bibs


def remove_curly_braces(text):
	result = []
	i = 0
	n = len(text)
	
	while i < n:
		if text[i:i + 2] == '{\\' or re.match(r".\{", text[i:i + 2]):
			# Found the start of a group we need to keep
			start = i
			i += 2  # Skip '{\'
			depth = 1  # Track nested braces
			
			while i < n and depth > 0:
				if text[i] == '{':
					depth += 1
				elif text[i] == '}':
					depth -= 1
				i += 1
			
			# Add the kept group to the result
			result.append(text[start:i])
		elif text[i] not in '{}':
			# Add any character that is not a curly brace
			result.append(text[i])
			i += 1
		else:
			# Skip pure curly braces
			i += 1
	
	return ''.join(result)

def clean_text(text):
	text = text.replace("‐", "-")  # Replace *ew* Mac *ew* hyphen with a proper hyphen
	text = re.sub(r"(?:\<\w+.*?\>)|(?:\<\/\w+\>)", "", text)  # Remove html tags
	# text = re.sub(r"\{(?!\\.\w\})|(?<!\{\\.\w)\}", r"", text)  # Remove { and }, but not if they are part of umlauts and accents
	# text = re.sub(r"(?<!\{\\.)\{(?!\\.)|(?<!\{\\..)\}", r"", text)  # Remove { and }, but not if they are part of umlauts and accents
	# text = re.sub(r"(?<!\{\\.)\{(?!\\)(.*?)\}", r"\1", text)  # Remove { and }, but not if they are part of umlauts and accents
	# text = re.sub(r"(?<!\\.)\{(?!\\)", r"", text)  # Remove { and }, but not if they are part of umlauts and accents
	# text = re.sub(r"(?<!.\{\w\}|\\.\w\})\}", r"", text)  # Remove { and }, but not if they are part of umlauts and accents
	# text = re.sub(r"(?<!\\.|.\\)\{(?!\\)|(?<!\{\\..|\\.\{.|.\{.\}|...\\)\}", r"", text)  # Remove { and }, but not if they are part of umlauts and accents
	
	# i = 0
	# while i < len(text) - 2:
	# 	if text[i] == "{" and text[i+1] != "\\":
	# 		if text
	# 		text = text[:i] + text[i+1:]
	# 	else:
	# 		i += 1
	
	text = text.replace("\n", " ")  # Replace newlines with spaces
	text = text.strip("\" ,")
	text = remove_curly_braces(text)
	text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
	return text.strip()  # Remove leading and trailing spaces


def update_discrepancies(bibs: dict, property_key: str, backup_property_key: str, bibname_new_old_discrepancy_dict: dict):
	print(f"New {property_key}s were found which were different from the originals:")
	print(tabulate([[i_bib, bib_name, original_property, new_property, bibs[bib_name].get("title", "(no title)"), bibs[bib_name].get("author", "(no author)")] for i_bib, [bib_name, [original_property, new_property]] in enumerate(bibname_new_old_discrepancy_dict.items())], headers=["#", "Bib name", f"Original {property_key}", f"New {property_key}", "Title", "Author"]))
	ans = input(f"List the # you want to update separated by \",\". [-1] for all. (The original values will be backed up as \"{backup_property_key}\"): ")
	ans = [x.strip() for x in ans.split(",")]
	if len(ans) == 1 and ans[0] == "":
		print("Nothing will be changed.")
	elif len(ans) == 1 and ans[0] == "-1":
		print("All will be updated.")
		bibs = add_prop_to_bib_entries(bibs, backup_property_key, {bib_name: original_property for bib_name, [original_property, new_property] in bibname_new_old_discrepancy_dict.items()})
		bibs = add_prop_to_bib_entries(bibs, property_key, {bib_name: new_property for bib_name, [original_property, new_property] in bibname_new_old_discrepancy_dict.items()}, True)
	else:
		ans = [int(x) for x in ans]
		if max(ans) > len(bibname_new_old_discrepancy_dict) -1:
			raise IndexError(f"Your answer includes a number ({max(ans)}) larger than the number of options ({len(bibname_new_old_discrepancy_dict) -1}).")
		
		to_update = {bib_name : value for bib_name, value in [list(bibname_new_old_discrepancy_dict.items())[i] for i in ans]}
		print("Updating:", ", ".join(to_update))
		bibs = add_prop_to_bib_entries(bibs, backup_property_key, {bib_name: original_property for bib_name, [original_property, new_property] in to_update.items()})
		bibs = add_prop_to_bib_entries(bibs, property_key, {bib_name: new_property for bib_name, [original_property, new_property] in to_update.items()}, True)
	return bibs



if False and __name__ == '__main__':
	bibtex_filename = "mythesislibrary_27JUNE2024.bib"
	bibs, bib_types = load_bibtex(bibtex_filename)
	
	latex_filename = "Thesis manuscript_27JUNE2024.tex"
	latex_file = load_file(latex_filename)
	citations, statements = latex2citations_statements(latex_file)
	
	print()
	
	# Look through citations and find discrepancies between bibs and citations
	# Any uppercase/lowercase discrepancies between bibtex and latex files
	case_discrepancies_bibtex_vs_latex = [[bib_name, citation] for bib_name in bibs for citation in citations if bib_name.lower() == citation.lower() and bib_name != citation]
	if case_discrepancies_bibtex_vs_latex:
		print("Some of your citations use different upper- and lowercase than the corresponding reference in the BibTeX:")
		print(tabulate([[bib_name, citation] for bib_name, citation in case_discrepancies_bibtex_vs_latex], headers=["Bib", "Citation"]))
		print("\nI've updated the BibTeX to match the case used in the LaTeX document.")
		for bib_name, citation_name in case_discrepancies_bibtex_vs_latex:  # Change citations to match bib names
			bibs[citation_name] = bibs.pop(bib_name)
			bib_types[citation_name] = bib_types.pop(bib_name)
			# index = citations.index(citation_name)
			# citations[index] = bib_name
	else:
		... # print("No upper/lowercase discrepancies between reference and corresponding citation.")
	
	
	discrepancies_latex_not_in_bibtex = list(set(citations).difference(set(bibs)))
	if discrepancies_latex_not_in_bibtex:
		print()
		print("Some citations are missing a reference in the BibTeX:")
		print("\n".join(discrepancies_latex_not_in_bibtex))
	else:
		... # print("No other discrepancies between citations and bibliography.")
	
	print()
	
	# Collect all the bibs which have been cited (now only working with this and not the full bibs!)
	bibs_in_citations = {bib_name: bib_entry for bib_name, bib_entry in bibs.items() if bib_name in citations}
	
	print(f"You have {len(citations)} ({len(set(citations))} unique) citations in the LaTeX document out of a total of {len(bibs)} references in the BibTeX ({round(len(set(citations)) / len(bibs) * 100, 1)}% of bibs used).")
	print("Only the references used as a citation will be kept.")
	print()
	
	# Get DOIs
	print("Finding missing DOIs from title and author...")
	DOIs = get_DOIs(bibs_in_citations, allow_copying_existing=True)
	# Add DOIs to bibs
	bibs_in_citations = add_prop_to_bib_entries(bibs_in_citations, "doi", DOIs)
	# Any mismatching DOIs? I.e. did get_DOIs() find better ones?
	DOI_discrepancies = {bib_name: [bib_entry["doi"], DOIs[bib_name]] for bib_name, bib_entry in bibs_in_citations.items() if "doi" in bib_entry and bib_entry["doi"] != DOIs[bib_name]}
	if DOI_discrepancies:
		bibs_in_citations = update_discrepancies(bibs_in_citations, "doi", "doi_orig", DOI_discrepancies)
	else:
		print("No new DOIs were found, which were different from the ones that were already there.")
	
	# Get PMIDs
	print("Getting PMIDs...")
	PMIDs = get_PMIDs(bibs_in_citations, allow_copying_existing=True)
	# Add PMIDs to bibs
	bibs_in_citations = add_prop_to_bib_entries(bibs_in_citations, "pmid", PMIDs)
	# Any mismatching PMIDs? I.e. did get_PMIDs() find better ones?
	PMID_discrepancies = {bib_name: [bib_entry["pmid"], PMIDs[bib_name]] for bib_name, bib_entry in bibs_in_citations.items() if "pmid" in bib_entry and bib_entry["pmid"] != PMIDs[bib_name]}
	if PMID_discrepancies:
		bibs_in_citations = update_discrepancies(bibs_in_citations, "pmid", "pmid_orig", PMID_discrepancies)
	else:
		print("No new PMIDs were found, which were different from the ones that were already there.")
	
	# Get reference_counts and citation_counts
	print("Getting reference counts and citation counts...")
	reference_counts, citation_counts = get_reference_and_citation_counts(bibs_in_citations, allow_copying_existing=True)
	# Add ref and cit counts to bibs
	bibs_in_citations = add_prop_to_bib_entries(bibs_in_citations, "reference_count", reference_counts)
	bibs_in_citations = add_prop_to_bib_entries(bibs_in_citations, "citation_count", citation_counts)
	# # Any mismatching ref or cit counts? I.e. did it find better ones?
	# PMID_discrepancies = {bib_name: [bib_entry["pmid"], PMIDs[bib_name]] for bib_name, bib_entry in bibs_in_citations.items() if "pmid" in bib_entry and bib_entry["pmid"] != PMIDs[bib_name]}
	# if PMID_discrepancies:
	# 	bibs_in_citations = update_discrepancies(bibs_in_citations, "pmid", "pmid_orig", PMID_discrepancies)
	
	# Clean up properties
	print("Cleaning up entries...")
	for key, entry in bibs.items():
		for field, value in entry.items():
			entry[field] = clean_text(value)
	
	# Get abstracts
	print("Getting abstracts...")
	abstracts = get_abstracts(bibs_in_citations, allow_copying_existing=True)
	# Add abstracts to bibs
	bibs_in_citations = add_prop_to_bib_entries(bibs_in_citations, "abstract", abstracts)
	# Any mismatching abstracts? I.e. did get_abstracts() find better ones?
	abstract_discrepancies = {bib_name: [bib_entry["abstract"], abstracts[bib_name]] for bib_name, bib_entry in bibs_in_citations.items() if "abstract" in bib_entry and bib_entry["abstract"] != abstracts[bib_name]}
	if abstract_discrepancies:
		bibs_in_citations = update_discrepancies(bibs_in_citations, "abstract", "abstract_orig", abstract_discrepancies)
	else:
		print("No differences between new and original abstracts.")
	
	# # Get abstracts, DOIs, PMIDs, reference and citation counts, and add them to the bib
	# bibs_in_citations = {bib_name : bib_entry for bib_name, bib_entry in bibs.items() if bib_name in citations}
	# abstracts, DOIs, PMIDs, reference_counts, citation_counts = run_go(bibs_in_citations, True, False)
	#
	# print("Adding new abstracts and PMIDs to bib entries...")
	# bibs = add_prop_to_bib_entries(bibs, "abstract", abstracts, False)
	# bibs = add_prop_to_bib_entries(bibs, "doi", DOIs)
	# bibs = add_prop_to_bib_entries(bibs, "pmid", PMIDs)
	# bibs = add_prop_to_bib_entries(bibs, "reference_count", reference_counts)
	# bibs = add_prop_to_bib_entries(bibs, "citation_count", citation_counts)
	
	
	# Find any close duplicates in the bibtex
	# find_matching_bibs(bibs_in_citations)
	DOIs_set = set([bib_entry["doi"] for bib_entry in bibs_in_citations.values() if "doi" in bib_entry])
	matching_DOIs_dict = {DOI : [bib_name for bib_name, bib_entry in bibs_in_citations.items() if "doi" in bib_entry and bib_entry["doi"] == DOI] for DOI in DOIs_set if sum([x.get("doi") == DOI for x in bibs_in_citations.values()]) > 1}
	N_matching_bibs = sum([len(x) for x in matching_DOIs_dict.values()])
	if N_matching_bibs:
		print()
		print(f"Some bib entries had exact matching DOIs:  {N_matching_bibs} / {len(bibs_in_citations)} ({round(N_matching_bibs / len(bibs_in_citations) * 100, 1)}%)")
		print(tabulate([[DOI, ", ".join(bib_names)] for DOI, bib_names in matching_DOIs_dict.items()], headers=["DOI", "Bibs"], maxcolwidths=[None, 100]))
	
	print()
	
	# Get a match score between statement in LaTeX file and the abstract(s) of the corresponding citation(s). To avoid bias, exclude common words (e.g. "the", "a"...)
	abstracts_without_error = {bib_name: bib_entry.get("abstract", "") if not bib_entry.get("abstract", "").startswith("ERROR:") else "" for bib_name, bib_entry in bibs_in_citations.items()}
	TF_IDF_scores = get_TF_IDF_scores(list(zip(citations, statements)), abstracts_without_error)
	BERT_scores = get_BERT_scores(list(zip(citations, statements)), abstracts_without_error)
	BioBERT_scores = get_BioBERT_scores(list(zip(citations, statements)), abstracts_without_error)
	overlap_scores = get_simple_overlap_scores(list(zip(citations, statements)), abstracts_without_error)
	# fuzzy_scores =
	
	# Save scores, bib_name, statement, abstract as a .csv file
	data = {
		"Overlap score (# common words / # of words in statement set)": overlap_scores,
		# "Fuzzy score": [get_fuzzy_score(str1, str2) for str1, str2 in zip()],  # TODO!!!
		# "TF_IDF score" : TF_IDF_scores,
		"BERT score": BERT_scores,
		"BioBERT score": BioBERT_scores,
		"bib name": citations,
		"Citation count": [citation_counts.get(bib_name, 0) for bib_name in citations],
		"Statement": statements,
		"Abstract": [abstracts_without_error.get(bib_name, "") for bib_name in citations],
		"Title": [bibs_in_citations[bib_name].get("title", "") if bib_name in bibs.keys() else "" for bib_name in citations],
		"DOI": [bibs_in_citations[bib_name].get("doi", "") if bib_name in bibs.keys() else "" for bib_name in citations],
		"PMID": [bibs_in_citations[bib_name].get("pmid", "") if bib_name in bibs.keys() else "" for bib_name in citations]
	}
	df = pd.DataFrame(data)
	df.to_csv("statement_vs_abstract_match_scores.csv", index=False, sep="\t")
	
	# Get a list of DOIs that don't have an abstract
	bibs_without_abstract = {bib_name: [bib_entry.get("doi"), bib_entry.get("pmid")] for bib_name, bib_entry in bibs_in_citations.items() if not bib_entry.get("abstract") or bib_entry.get("abstract").startswith("ERROR:")}
	print(f"\nBibs without abstract:  {len(bibs_without_abstract)} / {len(bibs_in_citations)} ({round(len(bibs_without_abstract) / len(bibs_in_citations) * 100, 1)}%)")
	print(tabulate([[bib_name, DOI, PMID] for bib_name, [DOI, PMID] in bibs_without_abstract.items()], headers=["Bib name", "DOI", "PMID"]))
	# print(f"Total count: {len(bibs_without_abstract)}/{len(bibs_in_citations)} ({round(len(bibs_without_abstract) / len(bibs_in_citations) * 100, 1)}%)")
	print()
	
	# Get a list of DOIs that don't have the number of citations
	bibs_without_citation_count = {bib_name: [bib_entry.get("doi"), bib_entry.get("pmid")] for bib_name, bib_entry in bibs_in_citations.items() if not bib_entry.get("citation_count")}
	print(f"\nBibs without citation count:  {len(bibs_without_citation_count)} / {len(bibs_in_citations)} ({round(len(bibs_without_citation_count) / len(bibs_in_citations) * 100, 1)}%)")
	print(tabulate([[bib_name, DOI, PMID] for bib_name, [DOI, PMID] in bibs_without_citation_count.items()], headers=["Bib name", "DOI", "PMID"]))
	# print(f"Total count: {len(bibs_without_citation_count)}/{len(bibs_in_citations)} ({round(len(bibs_without_citation_count) / len(bibs_in_citations) * 100, 1)}%)")
	print()

	# Get a list of DOIs that don't have the number of citations
	bibs_without_abstract_OR_citation_count = {bib_name: [bib_entry.get("doi"), bib_entry.get("pmid")] for bib_name, bib_entry in bibs_in_citations.items() if not bib_entry.get("abstract") or bib_entry.get("abstract").startswith("ERROR:") or not bib_entry.get("citation_count")}
	print(f"\nBibs without abstract or without citation count or without both:  {len(bibs_without_abstract_OR_citation_count)} / {len(bibs_in_citations)} ({round(len(bibs_without_abstract_OR_citation_count) / len(bibs_in_citations) * 100, 1)}%)")
	print(tabulate([[bib_name, DOI, PMID] for bib_name, [DOI, PMID] in bibs_without_abstract_OR_citation_count.items()], headers=["Bib name", "DOI", "PMID"]))
	# print(f"Total count: {len(bibs_without_abstract_OR_citation_count)}/{len(bibs_in_citations)} ({round(len(bibs_without_abstract_OR_citation_count) / len(bibs_in_citations) * 100, 1)}%)")
	print()

	# Get a list of bibs that don't have DOI
	bibs_without_DOI = {bib_name: [bib_entry.get("doi"), bib_entry.get("pmid")] for bib_name, bib_entry in bibs_in_citations.items() if not bib_entry.get("doi")}
	print(f"\nBibs without DOI:  {len(bibs_without_DOI)} / {len(bibs_in_citations)} ({round(len(bibs_without_DOI) / len(bibs_in_citations) * 100, 1)}%)")
	print(tabulate([[bib_name, DOI, PMID] for bib_name, [DOI, PMID] in bibs_without_DOI.items()], headers=["Bib name", "DOI", "PMID"]))
	# print(f"Total count: {len(bibs_without_DOI)}/{len(bibs_in_citations)} ({round(len(bibs_without_DOI) / len(bibs_in_citations) * 100, 1)}%)")
	
	# Save final bibtex file
	new_file = bibtex_filename.rsplit(".", 1)[0] + "_Fred.bib"
	save_bibtex(bibs_in_citations, bib_types, new_file)
	
	# Remove spelled out acronyms in LaTeX file. The first acronym should be spelled out, but the rest should just be the acronym.
	abbreviations = re.findall(r"((?:[\w-]+[^\w-]){,6})(\(.{2,7}\))((?:[^\w-][\w-]+){,6})", latex_file)
	print(tabulate([abbreviation for abbreviation in abbreviations]))
	latex_file_deacronymed = re.sub(r"\([\w\s\d]+\)", "", latex_file)
	
	
	# Plot density of bibs over year
	years_dict = {bib_name : int(bib_entry.get("year")) for bib_name, bib_entry in bibs_in_citations.items() if bib_entry.get("year")}
	years_list = [year for year in years_dict.values()]
	plt.figure(1, figsize=(7, 5))
	plt.hist(years_list, bins=max(years_list) - min(years_list) + 1)
	plt.xlabel("Publication year")
	plt.ylabel("# of references")
	plt.xticks(range((min(years_list) // 10) * 10, ((max(years_list) // 10) + 1) * 10 + 1, 10))
	plt.grid(which="major", axis="both")
	plt.tight_layout()
	plt.savefig("hist_bib_years.png")
	plt.show()

# TODO: Are there any close duplicates in the bibtex? Done..-
# TODO: Get a match score between statement in LaTeX file and the abstract of the corresponding citation. To avoid bias, exclude common words (e.g. "the", "a"...)-
# TODO: Get number of references for each bib entry. DONE.-

# TODO: Add title to csv-
# TODO: Check if title of DOI matches title of bibtex
# TODO: Don't overwrite if it's already there.-
# TODO: remove spelled out acronyms in parentheses
# TODO: no existing data is overwritten with new data: DOIs are found - even for ones where the DOI is wrong or smth, but it's not replaced in the bibtex........
# TODO: Make a density plot of bibs_in_citations and their year.
