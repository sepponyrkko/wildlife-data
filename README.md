# Data and code base used for the wildlife thesis

- notebooks/ - notebooks used for the thesis results and illustrations
  - *demo* - graphics that were created for illustrating the book
  - *tests* - scripts that were used for the thesis results
  - Run the notebooks using the "src" directory as the working directory
- src/ - the code base, "as is"
  - src/data - the analyzed data
     - q_cluster/ - GMM fitting snapshot of the clustering experiment
     - q_sep/  - trained fitting snapshot of the ICA tests
  - src/plots -  stored results from various tests in sample groups
     - delex - plots of the delexical feature space
     - delex - plots of the ICA lattice
- ontoreader/ - utility to read the database RDF into TSV files
- taxon_data_counts.ods - the counts of data: taxon level terms, pages and concepts


