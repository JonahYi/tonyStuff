import os
import hnswlib
import numpy as np
import unittest
import time
import matplotlib.pyplot as plt

class RandomlyDistributedQueries():
        def testCenteredQueries(standard_dev):
                dim = 768
                num_elements = 1000000
                #what is k?
                k = 10
                num_queries = 100

                # Generate sample data
                data = np.float32(np.random.normal(0, 1, size = [num_elements, dim]))

                # Declaring index
                hnsw_index = hnswlib.Index(space = 'ip', dim = dim)
                
                # Initiate hnsw index
                # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
                # during insertion of an element.
                # The capacity can be increased by saving/loading the index, see below.
                #
                # hnsw construction params:
                # ef_construction - controls index search speed/build speed tradeoff
                #
                # M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
                # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
                hnsw_index.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

                # Controlling the recall for hnsw by setting ef:
                # higher ef leads to better accuracy, but slower search
                hnsw_index.set_ef(200)
                
                print("Adding batch of %d elements" % (len(data)))
                hnsw_index.set_num_threads(32)
                hnsw_index.add_items(data)
                print("Indices built")
                
                # Generating query data
                query_data = np.float32(np.random.normal(0, standard_dev, size=[num_queries, dim]))

                #create an array here that keeps track of number of threads and runtime
                #also a for loop so that we can test for different numbers of threads
                num_threads = [1]
                for x in range(16):
                        num_threads.append(x * 2)
                throughput = []
                for threads in num_threads:
                        # Set number of threads used during batch search/construction in hnsw
                        # By default using all available cores
                        hnsw_index.set_num_threads(threads)

                        start_time = time.time()

                        # Query the elements:
                        labels_hnsw, distances_hnsw = hnsw_index.knn_query(query_data, k)

                        end_time = time.time()

                        #prints number of unique elements
                        print(np.unique(labels_hnsw[:,0]))

                        elapsed_time = end_time - start_time
                        throughput.append(num_queries/elapsed_time)
                np_num_threads = np.array(num_threads)
                np_throughput = np.array(throughput)

                plt.plot(np_num_threads, np_throughput)
                plt.xlabel("Num Threads")
                plt.ylabel("Throughput")
                plt.title("Standard Dev: " + str(standard_dev))
                plt.show()
                plt.savefig('std' + str(standard_dev) + '.png')

RandomlyDistributedQueries.testCenteredQueries(1)
RandomlyDistributedQueries.testCenteredQueries(0.5)
RandomlyDistributedQueries.testCenteredQueries(0.25)
RandomlyDistributedQueries.testCenteredQueries(0.125)