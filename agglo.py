'''
Reference : https://github.com/pramodsetlur/clustering
'''
import sys
import math
import heapq
import itertools
from helper.distance import euclidean
from helper.linkage import single_linkage, complete_linkage, average_group_linkage, average_linkage

DIMENSIONS = 0
POINTS_COUNT = 0
K_CLUSTERS = 0

def extract_eucledien_point(each_line):
    temp_line = each_line.strip().split(',')
    del temp_line[-1]
    temp_line = [float(i) for i in temp_line]
    return temp_line

def read_input_file(input_file):
    input_point_list = []
    with open(input_file) as file:
        for each_line in file:
            eucledian_point = extract_eucledien_point(each_line)
            input_point_list.append(eucledian_point)
    file.close()
    global DIMENSIONS
    DIMENSIONS = len(input_point_list[1])

    global POINTS_COUNT
    POINTS_COUNT = len(input_point_list)

    return input_point_list


def compute_centroid(cluster, input_point_list):
    cluster_size = len(cluster)
    cluster_euclidean_points = []
    centroid = []

    for i in range(0, DIMENSIONS):
        centroid.append(0.0)

    for i in range(0, cluster_size):
        point = cluster[i]
        euclean_points = input_point_list[point]
        cluster_euclidean_points.append(euclean_points)

    for i in range(0, DIMENSIONS):
        for j in range(0, cluster_size):
            centroid[i] += cluster_euclidean_points[j][i]
        centroid[i] /= cluster_size

    return centroid


def compute_eucledian_distance(clusterA, clusterB, input_point_list):
    sum = 0
    if 1 == len(clusterA):
        centroidA = input_point_list[clusterA[0]]
    else:
        centroidA = compute_centroid(clusterA, input_point_list)

    if 1 == len(clusterB):
        centroidB = input_point_list[clusterB[0]]
    else:
        centroidB = compute_centroid(clusterB, input_point_list)

    for i in range(DIMENSIONS):
        xi = centroidA[i]
        yi = centroidB[i]
        sum += (xi - yi) * (xi - yi)
    distance = math.sqrt(sum)

    return distance

def compute_pair_distance_add_to_heap(i, list, input_point_list, heap, linkage):
    j = i + 1
    clusterA = list[i]
    for k in range(j, len(list)):
        clusterB = list[k]
        
        if (linkage == 'single'):
            distance = single_linkage(clusterA, clusterB, input_point_list)
        elif (linkage == 'complete'):
            distance = complete_linkage(clusterA, clusterB, input_point_list)
        elif (linkage == 'average_group'):
            distance = average_group_linkage(clusterA, clusterB, input_point_list)
        elif (linkage == 'average'):
            distance = average_linkage(clusterA, clusterB, input_point_list)
        else:
            distance = compute_eucledian_distance(clusterA, clusterB, input_point_list)
        heap_item = [distance, [clusterA, clusterB]]
        heapq.heappush(heap, heap_item)

    return heap


def initialize_not_considered_list(not_considered_list):
    for i in range(POINTS_COUNT):
        not_considered_list.append([i])

    return not_considered_list

def setup(heap, input_point_list, linkage):
    not_considered_list = []
    not_considered_list = initialize_not_considered_list(not_considered_list)

    for i in range(POINTS_COUNT - 1):
        heap = compute_pair_distance_add_to_heap(i, not_considered_list, input_point_list, heap, linkage)

    return not_considered_list, heap

def check_heap(heap):
    sort = []
    while heap:
        sort.append(heapq.heappop(heap))
    for i in sort:
        print(i)

def copy_ncl_all_clusters(cluster_iteration, not_considered_list, all_clusters_dict):
    all_clusters_dict[cluster_iteration] = list(not_considered_list)
    return all_clusters_dict

def merge_clusters(clusterA, clusterB):
    return sorted(list(set(clusterA) | set(clusterB)))

def hierarchial_clustering(heap, input_point_list, linkage):
    all_clusters_dict = {}
    not_considered_list, heap = setup(heap, input_point_list, linkage)

    cluster_iteration = POINTS_COUNT
    all_clusters_dict = copy_ncl_all_clusters(cluster_iteration, not_considered_list, all_clusters_dict)

    while cluster_iteration > 1:
        min_distance_cluster = heapq.heappop(heap)
        distance = min_distance_cluster[0]
        cluster_information = min_distance_cluster[1]
        clusterA = cluster_information[0]
        clusterB = cluster_information[1]

        if clusterA in not_considered_list and clusterB in not_considered_list:
            not_considered_list.remove(clusterA)
            not_considered_list.remove(clusterB)
            merged_cluster = merge_clusters(clusterA, clusterB)
            not_considered_list.insert(0, merged_cluster)

            heap = compute_pair_distance_add_to_heap(0, not_considered_list, input_point_list, heap, linkage)
            cluster_iteration -= 1
            all_clusters_dict = copy_ncl_all_clusters(cluster_iteration, not_considered_list, all_clusters_dict)

    return all_clusters_dict

def setup_gold_standard(input_file):
    gold_standard_dict = {}
    with open(input_file) as file:
        point_number = 0
        for each_line in file:
            temp_line = each_line.strip().split(',')
            cluster_name = temp_line[-1]

            gold_standard_dict.setdefault(cluster_name, [])
            point_list = gold_standard_dict[cluster_name]
            point_list.append(point_number)
            gold_standard_dict[cluster_name] = point_list

            point_number += 1
    file.close()
    return gold_standard_dict

def compute_precision_recall(my_algo_pairs, gold_standard_pairs):
    intersection = set(my_algo_pairs).intersection(gold_standard_pairs)

    my_algo_count = float(len(my_algo_pairs))
    gold_standard_count = float(len(gold_standard_pairs))
    intersection_count = float(len(intersection))

    precision = intersection_count / my_algo_count
    recall = intersection_count / gold_standard_count

    return precision, recall

def find_pairs(pairs_list, points_list):
    temp_list = list(itertools.combinations(points_list, 2))
    pairs_list = pairs_list + temp_list
    return pairs_list

def gold_standard(input_file, k_clusters_list):
    gold_standard_dict = setup_gold_standard(input_file)
    gold_standard_pairs = []
    my_algo_pairs = []

    #Gold Standard pairs
    for cluster_name, points in gold_standard_dict.items():
        gold_standard_pairs = find_pairs(gold_standard_pairs, points)

    #My algo pairs
    for single_cluster in k_clusters_list:
        my_algo_pairs = find_pairs(my_algo_pairs, single_cluster)

    precision, recall = compute_precision_recall(my_algo_pairs, gold_standard_pairs)
    return precision, recall


def print_output(precision, recall, k_clusters):
    print(precision)
    print(recall)

    for each_cluster in k_clusters:
        print(each_cluster)

if __name__ == '__main__':

    if 4 != len(sys.argv):
        print("USAGE: python pramod_setlur_hclust.py [INPUT_FILE] [K_CLUSTERS] [LINKAGE_TYPE]")
    else:
        input_file = sys.argv[1]

        K_CLUSTERS = int(sys.argv[2])
        input_point_list = read_input_file(input_file)
        linkage_option = sys.argv[3]
        heap = []
        all_clusters_dict = hierarchial_clustering(heap, input_point_list, linkage_option)
        precision, recall = gold_standard(input_file, all_clusters_dict[K_CLUSTERS])
        print_output(precision, recall, all_clusters_dict[K_CLUSTERS])