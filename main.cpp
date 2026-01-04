#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <sstream>
#include <iomanip>

// --- TUNING PARAMETERS ---
const int BATCH_SIZE = 2048; 

using namespace std;

class OptimizedNQueens {
    int n;
    int limit_mask;
    unsigned long long total_solutions = 0;
    ofstream& outFile;
    double time_io_wait = 0.0;

public:
    OptimizedNQueens(int size, ofstream& out) : n(size), outFile(out) {
        limit_mask = (1 << n) - 1;
    }

    void solve() {
        // Use a local variable for reduction to avoid class member issues in some compilers
        unsigned long long global_count_accumulator = 0;

        #pragma omp parallel reduction(+:global_count_accumulator)
        {
            stringstream local_buffer;
            int buffer_fill_level = 0; 
            vector<int> path(n);

            // Dynamic scheduling balances the load
            #pragma omp for schedule(dynamic)
            for (int col = 0; col < n; ++col) {
                path[0] = col;
                
                // Track solutions specifically for this branch
                unsigned long long branch_count = 0;
                
                backtrack(1, (1 << col) << 1, (1 << col), (1 << col) >> 1, 
                          path, local_buffer, buffer_fill_level, branch_count);
                
                // Add this branch's findings to the thread's local accumulator
                global_count_accumulator += branch_count;
            }

            // Flush whatever is left in the buffer at the end
            if (buffer_fill_level > 0) {
                flush_buffer(local_buffer);
            }
        }
        
        // Update the class member with the final result
        total_solutions = global_count_accumulator;
    }

private:
    void flush_buffer(stringstream& buffer) {
        double start = omp_get_wtime();
        
        #pragma omp critical(disk_write)
        {
            outFile << buffer.rdbuf();
        }
        
        buffer.str(""); 
        buffer.clear();

        double end = omp_get_wtime();
        // Atomic update for stats (optional, slight overhead but safe)
        #pragma omp atomic
        time_io_wait += (end - start);
    }

    void backtrack(int row, int ld, int cols, int rd, 
                   vector<int>& path, 
                   stringstream& buffer, 
                   int& buffer_fill,
                   unsigned long long& branch_total) { // Added this counter
        
        if (row == n) {
            // 1. Increment the actual solution counter
            branch_total++;

            // 2. Format to buffer
            for (int i = 0; i < n; ++i) {
                buffer << (path[i] + 1) << (i == n - 1 ? "" : " ");
            }
            buffer << "\n";
            buffer_fill++;

            // 3. Flush if full
            if (buffer_fill >= BATCH_SIZE) {
                flush_buffer(buffer);
                buffer_fill = 0;
            }
            return;
        }

        int possible = ~(ld | cols | rd) & limit_mask;

        while (possible) {
            int bit = possible & -possible;
            possible -= bit;
            path[row] = __builtin_ctz(bit);
            backtrack(row + 1, (ld | bit) << 1, (cols | bit), (rd | bit) >> 1, 
                      path, buffer, buffer_fill, branch_total);
        }
    }
    
public:
    double get_io_overhead() { return time_io_wait; }
    unsigned long long get_total() { return total_solutions; }
};

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc != 2) {
        cerr << "Usage: ./nqueens_solver <input_file>\n";
        return 1;
    }

    string inputPath = argv[1];
    ifstream in(inputPath);
    if (!in) { cerr << "Error opening input file.\n"; return 1; }
    int n;
    in >> n;
    in.close();

    string outputPath = inputPath.substr(0, inputPath.find_last_of('.')) + "_output.txt";
    ofstream outFile(outputPath);
    if (!outFile) { cerr << "Error creating output file.\n"; return 1; }

    // Header Placeholder Trick
    outFile << n << "\n";
    long count_position = outFile.tellp();
    outFile << string(20, ' ') << "\n"; // Reserve space

    cout << "===========================================\n";
    cout << "   N-QUEENS OPTIMIZED SOLVER (N=" << n << ")\n";
    cout << "===========================================\n";
    cout << "CPU Threads      : " << omp_get_max_threads() << "\n";
    cout << "Batch Buffer Size: " << BATCH_SIZE << " solutions\n";
    cout << "Output Strategy  : Streamed (Low RAM usage)\n";
    cout << "-------------------------------------------\n";
    cout << "Solving... (This may take time for large N)\n";

    auto start_time = chrono::high_resolution_clock::now();

    OptimizedNQueens solver(n, outFile);
    solver.solve();

    auto end_time = chrono::high_resolution_clock::now();
    double total_seconds = chrono::duration<double>(end_time - start_time).count();

    // Fill in the correct count at the top of the file
    unsigned long long total = solver.get_total();
    outFile.seekp(count_position);
    outFile << total;
    outFile.close();

    cout << "-------------------------------------------\n";
    cout << "Processing Complete!\n";
    cout << "-------------------------------------------\n";
    cout << "Total Solutions  : " << total << "\n";
    cout << "Total Time       : " << fixed << setprecision(4) << total_seconds << " s\n";
    
    if (total_seconds > 0)
        cout << "Approx Throughput: " << (unsigned long long)(total / total_seconds) << " solutions/sec\n";
    
    cout << "Output File      : " << outputPath << "\n";
    cout << "===========================================\n";

    return 0;
}