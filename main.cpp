#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <sstream>
#include <iomanip>

using namespace std;

// --- CONFIGURATION & SAFETY ---
const int BATCH_SIZE = 400000;  // High RAM usage (better speed)
const int MAX_WRITE_N = 18;     // Safety: Do not write file if N > 17 (Prevents 100GB+ files)
const int HARD_LIMIT_N = 20;    // Hard Stop: Do not run at all if N > 20

class OptimizedNQueens {
    int n;
    int limit_mask;
    unsigned long long total_solutions = 0;
    ofstream* outFile; // Pointer allows us to pass "nullptr" to disable writing
    double time_io_wait = 0.0;
    bool write_to_disk;

public:
    // If "out" is nullptr, we run in Count-Only mode
    OptimizedNQueens(int size, ofstream* out) : n(size), outFile(out) {
        limit_mask = (1 << n) - 1;
        write_to_disk = (outFile != nullptr);
    }

    void solve() {
        unsigned long long global_count_accumulator = 0;

        #pragma omp parallel reduction(+:global_count_accumulator)
        {
            stringstream local_buffer;
            int buffer_fill_level = 0; 
            vector<int> path(n);

            #pragma omp for schedule(dynamic)
            for (int col = 0; col < n; ++col) {
                path[0] = col;
                unsigned long long branch_count = 0;
                
                backtrack(1, (1 << col) << 1, (1 << col), (1 << col) >> 1, 
                          path, local_buffer, buffer_fill_level, branch_count);
                
                global_count_accumulator += branch_count;
            }

            // Flush remaining buffer (Only if writing is enabled)
            if (write_to_disk && buffer_fill_level > 0) {
                flush_buffer(local_buffer);
            }
        }
        total_solutions = global_count_accumulator;
    }

private:
    void flush_buffer(stringstream& buffer) {
        if (!write_to_disk) return; // Double check

        double start = omp_get_wtime();
        #pragma omp critical(disk_write)
        {
            (*outFile) << buffer.rdbuf();
        }
        buffer.str(""); 
        buffer.clear();
        double end = omp_get_wtime();
        
        #pragma omp atomic
        time_io_wait += (end - start);
    }

    void backtrack(int row, int ld, int cols, int rd, 
                   vector<int>& path, 
                   stringstream& buffer, 
                   int& buffer_fill,
                   unsigned long long& branch_total) {
        
        if (row == n) {
            branch_total++;

            // OPTIMIZATION: If we are not writing to disk, return immediately!
            // This makes N>17 runs extremely fast.
            if (!write_to_disk) return;

            for (int i = 0; i < n; ++i) {
                buffer << (path[i] + 1) << (i == n - 1 ? "" : " ");
            }
            buffer << "\n";
            buffer_fill++;

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

    // --- SAFETY CHECKS ---
    if (n > HARD_LIMIT_N) {
        cerr << "ERROR: N=" << n << " is too large for this solver (Limit: " << HARD_LIMIT_N << "). Exiting.\n";
        return 1;
    }

    // Determine Mode
    bool safe_to_write = (n <= MAX_WRITE_N);
    string outputPath = inputPath.substr(0, inputPath.find_last_of('.')) + "_output.txt";
    
    // We strictly control the ofstream here
    ofstream* filePtr = nullptr;
    ofstream outFile;

    if (safe_to_write) {
        outFile.open(outputPath);
        if (!outFile) { cerr << "Error creating output file.\n"; return 1; }
        
        // Header Placeholder Trick
        outFile << n << "\n";
        outFile << string(20, ' ') << "\n"; 
        filePtr = &outFile;
    }

    // --- DASHBOARD ---
    cout << "===========================================\n";
    cout << "   N-QUEENS SAFETY SOLVER (N=" << n << ")\n";
    cout << "===========================================\n";
    cout << "CPU Threads      : " << omp_get_max_threads() << "\n";
    cout << "Mode             : " << (safe_to_write ? "FULL WRITE (Solutions + Count)" : "COUNT ONLY (Disk Safety Active)") << "\n";
    if (safe_to_write) cout << "Batch Buffer     : " << BATCH_SIZE << " solutions\n";
    cout << "-------------------------------------------\n";
    cout << "Solving...\n";

    auto start_time = chrono::high_resolution_clock::now();

    // Pass "nullptr" if we shouldn't write, automatically disabling I/O logic
    OptimizedNQueens solver(n, filePtr);
    solver.solve();

    auto end_time = chrono::high_resolution_clock::now();
    double total_seconds = chrono::duration<double>(end_time - start_time).count();
    unsigned long long total = solver.get_total();

    // --- FINALIZE OUTPUT ---
    if (safe_to_write) {
        outFile.seekp(to_string(n).length() + 1); // Jump to line 2 (after N + newline)
        outFile << total;
        outFile.close();
    } else {
        // For unsafe N, we might still want to save just the number to a file
        ofstream summaryFile(outputPath);
        summaryFile << "N=" << n << "\n";
        summaryFile << "Total=" << total << "\n";
        summaryFile << "(Board output suppressed due to size safety limit > " << MAX_WRITE_N << ")\n";
        summaryFile.close();
    }

    cout << "-------------------------------------------\n";
    cout << "Processing Complete!\n";
    cout << "-------------------------------------------\n";
    cout << "Total Solutions  : " << total << "\n";
    cout << "Total Time       : " << fixed << setprecision(4) << total_seconds << " s\n";
    if (total_seconds > 0)
        cout << "Approx Throughput: " << (unsigned long long)(total / total_seconds) << " solutions/sec\n";
    cout << "Output File      : " << outputPath << "\n";
    if (!safe_to_write) cout << "(Note: Output file contains COUNT ONLY to save disk space)\n";
    cout << "===========================================\n";

    return 0;
}