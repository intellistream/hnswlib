#include "hnswlib/hnswlib.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <ctime>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <cstdlib>

struct ProgramOptions {
    std::string base_path = "sift_base.bin";
    std::string query_path = "sift_query_1k.bin";
    std::string index_output = "hnsw_sift.bin";
    std::string pruning_output = "hnsw_pruning_analysis.txt";
    std::string query_log = "hnsw_query_log.csv";
    int M = 24;
    int ef_construction = 200;
    int ef_search = 40;
    int k = 10;
    int top_pruned = 1000;
    int top_queries = 100;
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --base <path>            Path to base dataset (default: sift_base.bin)\n"
              << "  --query <path>           Path to query dataset (default: sift_query_1k.bin)\n"
              << "  --M <int>                Maximum number of outgoing connections (default: 24)\n"
              << "  --ef-construction <int>  Construction ef parameter (default: 200)\n"
              << "  --ef-search <int>        Search ef parameter (default: 40)\n"
              << "  --k <int>                Number of nearest neighbors to retrieve (default: 10)\n"
              << "  --index-out <path>       Path to save built index (default: hnsw_sift.bin)\n"
              << "  --pruning-out <path>     Path to pruning analysis output (default: hnsw_pruning_analysis.txt)\n"
              << "  --query-log <path>       Path to rolling query log CSV (default: hnsw_query_log.csv)\n"
              << "  --top-pruned <int>       Number of top pruned nodes to capture (default: 1000)\n"
              << "  --top-queries <int>      Number of top queries to record (default: 100)\n"
              << "  -h, --help               Show this help message and exit\n";
}

ProgramOptions parseArguments(int argc, char** argv) {
    ProgramOptions opts;

    auto requireValue = [&](int& index, const std::string& flag) -> std::string {
        if (index + 1 >= argc) {
            std::cerr << "Missing value for option: " << flag << std::endl;
            printUsage(argv[0]);
            std::exit(1);
        }
        return std::string(argv[++index]);
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--base") {
            opts.base_path = requireValue(i, arg);
        } else if (arg == "--query") {
            opts.query_path = requireValue(i, arg);
        } else if (arg == "--M" || arg == "--m") {
            opts.M = std::stoi(requireValue(i, arg));
        } else if (arg == "--ef-construction") {
            opts.ef_construction = std::stoi(requireValue(i, arg));
        } else if (arg == "--ef-search") {
            opts.ef_search = std::stoi(requireValue(i, arg));
        } else if (arg == "--k") {
            opts.k = std::stoi(requireValue(i, arg));
        } else if (arg == "--index-out") {
            opts.index_output = requireValue(i, arg);
        } else if (arg == "--pruning-out") {
            opts.pruning_output = requireValue(i, arg);
        } else if (arg == "--query-log") {
            opts.query_log = requireValue(i, arg);
        } else if (arg == "--top-pruned") {
            opts.top_pruned = std::stoi(requireValue(i, arg));
        } else if (arg == "--top-queries") {
            opts.top_queries = std::stoi(requireValue(i, arg));
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    return opts;
}

std::vector<float> readSiftData(const std::string& filename, int& num_vectors, int& dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }
    
    file.read(reinterpret_cast<char*>(&num_vectors), sizeof(int));
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    
    std::cout << "File: " << filename << std::endl;
    std::cout << "Number of vectors: " << num_vectors << ", Dimension: " << dim << std::endl;
    
    std::vector<float> data(num_vectors * dim);
    file.read(reinterpret_cast<char*>(data.data()), num_vectors * dim * sizeof(float));
    file.close();
    
    return data;
}

struct QueryPruningInfo {
    int query_id;
    long pruned_nodes_visited;
    std::vector<unsigned int> visited_pruned_nodes;
};

void writePruningResults(const std::string& filename, 
                        const std::vector<std::pair<unsigned int, long>>& top_pruned_nodes,
                        const std::vector<QueryPruningInfo>& top_queries,
                        int requested_pruned_count,
                        int requested_query_count) {
    std::ofstream log_file(filename);
    if (!log_file.is_open()) {
        std::cerr << "Cannot open log file: " << filename << std::endl;
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    log_file << "# HNSW Pruning Analysis Results" << std::endl;
    log_file << "# Generated at: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;
    log_file << std::endl;
    
    // Write pruned nodes
    log_file << "[TOP_" << requested_pruned_count << "_PRUNED_NODES]" << std::endl;
    int pruned_to_write = std::min(static_cast<int>(top_pruned_nodes.size()), requested_pruned_count);
    for (int i = 0; i < pruned_to_write; ++i) {
        const auto& node = top_pruned_nodes[i];
        log_file << node.first << ":" << node.second << std::endl;
    }
    
    log_file << std::endl;
    log_file << "[TOP_" << requested_query_count << "_QUERIES_VISITING_PRUNED_NODES]" << std::endl;
    log_file << "# Format: query_id:pruned_nodes_visited:visited_pruned_node_ids" << std::endl;
    
    // Write top 100 queries
    int queries_to_write = std::min(static_cast<int>(top_queries.size()), requested_query_count);
    for (int i = 0; i < queries_to_write; ++i) {
        const auto& query = top_queries[i];
        log_file << query.query_id << ":" << query.pruned_nodes_visited << ":";
        for (size_t j = 0; j < query.visited_pruned_nodes.size(); j++) {
            log_file << query.visited_pruned_nodes[j];
            if (j < query.visited_pruned_nodes.size() - 1) {
                log_file << ",";
            }
        }
        log_file << std::endl;
    }
    
    log_file.close();
    std::cout << "Results written to: " << filename << std::endl;
}

void appendQueryIndexToLog(const std::string& filename,
                           const std::vector<QueryPruningInfo>& top_queries,
                           const ProgramOptions& opts) {
    std::ofstream log_file(filename, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Cannot open log file: " << filename << std::endl;
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    log_file << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
             << "," << top_queries.size()
             << "," << opts.M
             << "," << opts.ef_construction
             << "," << opts.ef_search
             << "," << opts.k
             << ",\"" << opts.base_path << "\""
             << ",\"" << opts.query_path << "\"";
    
    for (const auto& query : top_queries) {
        log_file << "," << query.query_id << ":" << query.pruned_nodes_visited;
    }
    log_file << std::endl;
    log_file.close();
}

std::vector<QueryPruningInfo> findQueriesWithMostPrunedNodes(hnswlib::HierarchicalNSW<float>* alg_hnsw,
                                                           const std::vector<float>& query_data,
                                                           int query_num_vectors, int query_dim, int k,
                                                           const std::vector<std::pair<unsigned int, long>>& top_pruned_nodes,
                                                           int top_queries) {
    std::cout << "\nAnalyzing queries that visit most pruned nodes..." << std::endl;
    
    // Create a set of pruned node IDs for fast lookup
    std::unordered_set<unsigned int> pruned_node_set;
    for (const auto& node : top_pruned_nodes) {
        pruned_node_set.insert(node.first);
    }
    
    std::vector<QueryPruningInfo> query_info;
    
    // Analyze each query
    for (int i = 0; i < query_num_vectors; i++) {
        // Reset visit counts for this query
        alg_hnsw->resetSearchPathStats();
        
        // Perform search
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = 
            alg_hnsw->searchKnn(query_data.data() + i * query_dim, k);
        
        // Count how many pruned nodes were visited
        long pruned_visited = 0;
        std::vector<unsigned int> visited_pruned;
        
        for (size_t j = 0; j < alg_hnsw->getCurrentElementCount(); j++) {
            long visit_count = alg_hnsw->getNodeVisitCount(j);
            if (visit_count > 0 && pruned_node_set.find(j) != pruned_node_set.end()) {
                pruned_visited += visit_count;
                visited_pruned.push_back(j);
            }
        }
        
        if (pruned_visited > 0) {
            query_info.push_back({i, pruned_visited, visited_pruned});
        }
        
        if ((i + 1) % 100 == 0) {
            std::cout << "Processed " << (i + 1) << " queries..." << std::endl;
        }
    }
    
    // Sort by number of pruned nodes visited
    std::sort(query_info.begin(), query_info.end(),
              [](const QueryPruningInfo& a, const QueryPruningInfo& b) {
                  return a.pruned_nodes_visited > b.pruned_nodes_visited;
              });
    
    // Return top queries
    if (query_info.size() > top_queries) {
        query_info.resize(top_queries);
    }
    
    std::cout << "Found " << query_info.size() << " queries that visit pruned nodes" << std::endl;
    
    return query_info;
}

int main(int argc, char** argv) {
    ProgramOptions opts = parseArguments(argc, argv);

    if (opts.M <= 0 || opts.ef_construction <= 0 || opts.ef_search <= 0 || opts.k <= 0) {
        std::cerr << "Error: M, ef_construction, ef_search, and k must be positive integers." << std::endl;
        return 1;
    }

    if (opts.top_pruned <= 0 || opts.top_queries <= 0) {
        std::cerr << "Error: top-pruned and top-queries must be positive integers." << std::endl;
        return 1;
    }

    std::cout << "HNSW Library Experiment Runner" << std::endl;
    std::cout << "Base dataset: " << opts.base_path << std::endl;
    std::cout << "Query dataset: " << opts.query_path << std::endl;
    std::cout << "Parameters -> M: " << opts.M
              << ", ef_construction: " << opts.ef_construction
              << ", ef_search: " << opts.ef_search
              << ", k: " << opts.k << std::endl;

    std::ifstream check_file(opts.query_log);
    if (!check_file.good()) {
        std::ofstream header_file(opts.query_log);
        header_file << "timestamp,query_count,M,ef_construction,ef_search,k,base_dataset,query_dataset,top_query_indices" << std::endl;
        header_file.close();
        std::cout << "Created query log file: " << opts.query_log << std::endl;
    }
    check_file.close();

    int base_num_vectors = 0;
    int base_dim = 0;
    std::vector<float> base_data = readSiftData(opts.base_path, base_num_vectors, base_dim);

    int query_num_vectors = 0;
    int query_dim = 0;
    std::vector<float> query_data = readSiftData(opts.query_path, query_num_vectors, query_dim);

    if (base_dim != query_dim) {
        std::cerr << "Error: Base dataset and query dataset dimensions do not match!" << std::endl;
        return 1;
    }

    if (base_num_vectors == 0 || query_num_vectors == 0) {
        std::cerr << "Error: Dataset is empty." << std::endl;
        return 1;
    }

    hnswlib::L2Space space(base_dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw =
        new hnswlib::HierarchicalNSW<float>(&space, base_num_vectors, opts.M, opts.ef_construction, 1);

    alg_hnsw->enablePruningStats(true);
    alg_hnsw->enableSearchPathTracking(true);

    std::cout << "\nStarting index construction..." << std::endl;

    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " OpenMP threads for construction" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < base_num_vectors; i++) {
        alg_hnsw->addPoint(base_data.data() + i * base_dim, i);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Index construction completed, time: " << build_duration.count() << " ms" << std::endl;

    alg_hnsw->setEf(opts.ef_search);

    std::cout << "\nStarting search test..." << std::endl;
    std::cout << "Using " << num_threads << " OpenMP threads for search" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<hnswlib::labeltype>> results(query_num_vectors);

    #pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < query_num_vectors; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw->searchKnn(query_data.data() + i * query_dim, opts.k);

        std::vector<hnswlib::labeltype> query_results;
        while (!result.empty()) {
            query_results.push_back(result.top().second);
            result.pop();
        }
        results[i] = query_results;

        if (omp_get_thread_num() == 0 && (i + 1) % 100 == 0) {
            std::cout << "Processed " << (i + 1) << " queries..." << std::endl;
        }
    }

    end_time = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Search completed!" << std::endl;
    std::cout << "Number of queries: " << query_num_vectors << std::endl;
    const auto total_ms = static_cast<double>(search_duration.count());
    std::cout << "Total search time: " << total_ms << " ms" << std::endl;
    if (total_ms > 0.0) {
        std::cout << "Average time per query: " << total_ms / query_num_vectors << " ms" << std::endl;
        std::cout << "Queries per second (QPS): " << (query_num_vectors * 1000.0) / total_ms << std::endl;
    } else {
        std::cout << "Average time per query: N/A (zero duration)" << std::endl;
        std::cout << "Queries per second (QPS): N/A (zero duration)" << std::endl;
    }

    std::cout << "\nResults of first 3 queries:" << std::endl;
    for (int i = 0; i < std::min(3, query_num_vectors); i++) {
        std::cout << "Query " << i << " nearest neighbors: ";
        for (int j = 0; j < std::min(opts.k, static_cast<int>(results[i].size())); j++) {
            std::cout << results[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nSaving index to file..." << std::endl;
    alg_hnsw->saveIndex(opts.index_output);
    std::cout << "Index saved to: " << opts.index_output << std::endl;

    std::cout << "\nPruning Statistics:" << std::endl;
    auto top_pruned = alg_hnsw->getTopPrunedNodes(opts.top_pruned);
    if (top_pruned.empty()) {
        std::cout << "No pruning occurred during construction." << std::endl;
    } else {
        std::cout << "Top " << opts.top_pruned << " most pruned nodes collected. Actual count: " << top_pruned.size() << std::endl;
    }

    // Find queries that visit most pruned nodes
    auto top_queries = findQueriesWithMostPrunedNodes(alg_hnsw, query_data, query_num_vectors, query_dim, opts.k, top_pruned, opts.top_queries);

    // Write detailed results to file
    writePruningResults(opts.pruning_output, top_pruned, top_queries, opts.top_pruned, opts.top_queries);

    // Also append query indices to CSV log
    appendQueryIndexToLog(opts.query_log, top_queries, opts);

    delete alg_hnsw;

    std::cout << "\nProgram execution completed!" << std::endl;
    return 0;
}
