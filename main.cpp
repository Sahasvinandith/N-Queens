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
// How many solutions a thread holds in RAM before writing to disk.
// 2048 is a sweet spot: Large enough to reduce lock contention, small enough to save RAM.
const int BATCH_SIZE = 2048; 

using namespace std;

class OptimizedNQueens {
    int n;
    int limit_mask;
    unsigned long long total_solutions = 0;
    ofstream& outFile;
    
    // Timer for calculation vs I/O tracking (approximate)
    double time_io_wait = 0.0;

public:
    OptimizedNQueens(int size, ofstream& out) : n(size), outFile(out) {
        limit_mask = (1 << n) - 1;
    }

    void solve() {
        // Start parallel region
        #pragma omp parallel
        {
            // --- THREAD LOCAL STORAGE ---
            // Each thread gets its own private "notepad" (stringstream).
            // This prevents threads from fighting over memory.
            stringstream local_buffer;
            int buffer_count = 0; 
            vector<int> path(n); // Stores current board state

            // --- DYNAMIC SCHEDULING ---
            // 'dynamic' ensures that if one thread finishes a hard branch,
            // it immediately grabs a new chunk of work. No core sits idle.
            #pragma omp for schedule(dynamic) reduction(+:total_solutions)
            for (int col = 0; col < n; ++col) {
                path[0] = col;
                
                // Start recursive solver for this column's branch
                backtrack(1, (1 << col) << 1, (1 << col), (1 << col) >> 1, 
                          path, local_buffer, buffer_count);
            }

            // --- FINAL FLUSH ---
            // Write any remaining solutions currently in the buffer
            if (buffer_count > 0) {
                flush_buffer(local_buffer);
            }
        }
    }

private:
    // Helper to write buffer to disk safely
    void flush_buffer(stringstream& buffer) {
        // We time how long we wait for the disk
        double start = omp_get_wtime();
        
        // CRITICAL SECTION: Only one thread can write to the file at a time.
        // Because we buffer 2048 solutions, we enter this section 2048x LESS often.
        #pragma omp critical(disk_write)
        {
            outFile << buffer.rdbuf();
        }
        
        // Clear the buffer for reuse
        buffer.str(""); 
        buffer.clear();

        // Track stats
        double end = omp_get_wtime();
        #pragma omp atomic
        time_io_wait += (end - start);
    }

    // Core Bitwise Backtracking
    void backtrack(int row, int ld, int cols, int rd, 
                   vector<int>& path, 
                   stringstream& buffer, 
                   int& count) {
        
        if (row == n) {
            // 1. Format solution into local RAM buffer
            for (int i = 0; i < n; ++i) {
                buffer << (path[i] + 1) << (i == n - 1 ? "" : " ");
            }
            buffer << "\n";
            count++;

            // 2. Check if buffer is full
            if (count >= BATCH_SIZE) {
                flush_buffer(buffer);
                count = 0;
            }
            return;
        }

        // Bitwise magic to find safe spots
        int possible = ~(ld | cols | rd) & limit_mask;

        while (possible) {
            int bit = possible & -possible; // Extract lowest bit (rightmost safe column)
            possible -= bit;                // Remove it from 'possible' set
            
            path[row] = __builtin_ctz(bit); // Convert bit to index (0-based) for printing

            // Recursive call: Shift diagonals for next row
            backtrack(row + 1, (ld | bit) << 1, (cols | bit), (rd | bit) >> 1, 
                      path, buffer, count);
        }
    }
    
public:
    double get_io_overhead() { return time_io_wait; }
    unsigned long long get_total() { return total_solutions; }
};

int main(int argc, char** argv) {
    // Optimizes C++ standard streams
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc != 2) {
        cerr << "Usage: ./nqueens_final <input_file>\n";
        return 1;
    }

    // 1. Read Input
    string inputPath = argv[1];
    ifstream in(inputPath);
    if (!in) { cerr << "Error opening input file.\n"; return 1; }
    int n;
    in >> n;
    in.close();

    // 2. Prepare Output
    string outputPath = inputPath.substr(0, inputPath.find_last_of('.')) + "_output.txt";
    ofstream outFile(outputPath);
    if (!outFile) { cerr << "Error creating output file.\n"; return 1; }

    // --- HEADER PATCHING TRICK ---
    // We write N, then leave a blank space for the Total Count.
    // We will come back and fill this in later.
    outFile << n << "\n";
    long count_position = outFile.tellp(); // Remember this position
    outFile << string(20, ' ') << "\n";    // Reserve 20 spaces (enough for 64-bit int)

    // 3. Print Stats to Console
    cout << "===========================================\n";
    cout << "   N-QUEENS OPTIMIZED SOLVER (N=" << n << ")\n";
    cout << "===========================================\n";
    cout << "CPU Threads      : " << omp_get_max_threads() << "\n";
    cout << "Batch Buffer Size: " << BATCH_SIZE << " solutions\n";
    cout << "Output Strategy  : Streamed (Low RAM usage)\n";
    cout << "-------------------------------------------\n";
    cout << "Solving... (This may take time for large N)\n";

    // 4. Run Solver
    auto start_time = chrono::high_resolution_clock::now();

    OptimizedNQueens solver(n, outFile);
    solver.solve();

    auto end_time = chrono::high_resolution_clock::now();
    double total_seconds = chrono::duration<double>(end_time - start_time).count();

    // 5. Finalize File (The "Rewind")
    unsigned long long total = solver.get_total();
    outFile.seekp(count_position); // Jump back to line 2
    outFile << total;              // Overwrite the spaces with the real number
    outFile.close();

    // 6. Final Report
    double io_overhead = solver.get_io_overhead(); // Total CPU time spent waiting for lock
    // Note: IO overhead is summed across threads, so it can be > wall time.
    // We normalize it by thread count for an estimate, or just show raw.
    
    cout << "-------------------------------------------\n";
    cout << "Processing Complete!\n";
    cout << "-------------------------------------------\n";
    cout << "Total Solutions  : " << total << "\n";
    cout << "Total Time       : " << fixed << setprecision(4) << total_seconds << " s\n";
    cout << "Approx Throughput: " << (total / total_seconds) << " solutions/sec\n";
    cout << "Output File      : " << outputPath << "\n";
    cout << "===========================================\n";

    return 0;
}