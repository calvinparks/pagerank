import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    

    # dictionary to return the probabilites of each page being visited depending on the 'page' the user is on.
    result_transition_corpus = {}

    #find eacn page in the corpus and assign a default probibility that a user will click on each next.
    for corpus_page, links in corpus.items():
        result_transition_corpus[corpus_page] = ((1-damping_factor) / len(corpus))

    #find eacn next-page in the links set for the give 'page' the the user is on calculate the probability that a user will click on each next-pageof them.
    for corpus_page, links in corpus.items():
        if corpus_page == page:
            for link in links:
                result_transition_corpus[link] = ((1-damping_factor) / len(corpus)) + (damping_factor/len(links))
    
    
    return result_transition_corpus
    
   


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #create dictionary to track the number of page visits per corpus key
    page_click_count_results = {}

    # used to track list of all pages in the corpus
    all_page_list=[]


    #initialize to all zero's the dictionary to track the number of page visits per corpus key
    for corpus_page, links in corpus.items():
        page_click_count_results[corpus_page] = 0
        all_page_list.append(corpus_page)


    # list to track probabilites of each page in all_page_list
    all_page_probability_list = []
  

    # determine the probabilites for each page
    for i in range(len(all_page_list)):
        all_page_probability_list.append(1/len(all_page_list))


    #randomly select 1st page using equal probabilities 
    equal_probabilities = [1/len(all_page_list)] * len(all_page_list)
    next_page = random.choices(all_page_list, weights = equal_probabilities)[0]

    #get transition model for 1st page
    transiton_results = transition_model(corpus, next_page, damping_factor)





    #loop 'n' times to sample 
    for i in range(n, 0, -1):
        all_page_probability_list = []
       
        # refresh the all_page_probability_list with new probabilites returned from transition_model() 
        for corpus_page, pages_probability in transiton_results.items():
            all_page_probability_list.append(pages_probability)
 

        #choose the next page based on the transiton model probabilities
        next_page = random.choices(all_page_list, weights = all_page_probability_list)
        next_page = next_page[0] # force list to string
        page_click_count_results[next_page] += (1/n)

        #get transition model for next page
        transiton_results = transition_model(corpus, next_page, damping_factor)


    return page_click_count_results
    


    


def create_corpus_links_matrix(corpus):
    # Create a Dictionary of copus nodes(key) and assign each an order tracking number.
    # These order tracking numbers  will later be used to assign probabilities each to row and colum positions in a matrix.
   
    corpus_matrix_position_dictionary = {}  # used to hold a matrix position for each corpus node
    corpus_matrix_link_count_dictionary = {} # tracks the number of links that each corpus node has
    
    corpus_matrix_position = -1
    for key, value in corpus.items():
      corpus_matrix_position +=1
      corpus_matrix_position_dictionary[key] = corpus_matrix_position
      corpus_matrix_link_count_dictionary[key] = len(value) # THis is the number of links on the current page


    #create matrix initaialized to all 0's
    corpus_matrix = np.zeros((len(corpus),len(corpus)))

    # populate matrix columns and rows with probilities
    # the key will hold the name be each corpus pages
    for key, value in corpus_matrix_position_dictionary.items():
        #print(key,value,len(corpus[key]))

        # if the corpus page has links then figure out the probibilites to each page and populate the matrix's row/colums appropriately
        for the_number_of_links in range(len(corpus[key])):
            page_links = corpus.get(key)
            number_of_page_links = len(page_links)

            if number_of_page_links > 0:
                for dummy_x in range(number_of_page_links):
                    #print(page_links.pop())
                    # get the page-name (this will be used as a key to the appropiate corpus_matrix_position_dictionary's column)
                    page_link_matrix_key = page_links.pop()
                    corpus_matrix[corpus_matrix_position_dictionary[key], corpus_matrix_position_dictionary[page_link_matrix_key]] = 1/number_of_page_links
                   
    # swap the rows with the colums because the probability matrix will be a single column of values
    corpus_matrix = corpus_matrix.T

    # check if A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself
    corpus_links_matrix_sum=corpus_matrix.sum(axis=0) # add up all values in each column
    all_pages_link_ratio = 1/len(corpus_links_matrix_sum) # determine the link ration for all pages
    for x in range(len(corpus_links_matrix_sum)):
        # if the column values for all possible links is zero then this page eighter links to it's self or is recursive(per specification)
        if corpus_links_matrix_sum[x] == 0.0: 
            corpus_matrix[:, x] = all_pages_link_ratio
    
    return corpus_matrix
   





def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    # find the number of pages in the corpus
    corpus_page_count=len(corpus)

    #initialize a matrix to be used for link probabilites to 1
    probilites_matrix = np.ones([corpus_page_count, 1])
    
    # create a corpus that tracks all out going links on each page
    corpus_links_matrix = create_corpus_links_matrix(corpus)
    
    rank = 1.0 / corpus_page_count * probilites_matrix

    for i in range(10):
        rank = damping_factor * np.dot(corpus_links_matrix,rank) + ((1 - damping_factor) * 1.0 / corpus_page_count * probilites_matrix)  
    

   
    # create dictionary of the results.  key is page name, value is the rank value
    ii = -1
    iteration_results = {}
    for key in corpus.keys() :
        ii+=1
        a = key
        if ii < len(corpus):
            results_item = rank[ii,:]  # get the array of the column values(there is only 1 value in the array)
            iteration_results[key]=results_item[0] # save key value to the dictionary
            
    
    return iteration_results


if __name__ == "__main__":
    main()
