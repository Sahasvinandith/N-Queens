#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <iomanip>

// Strategy Enum
enum Mode {
    FIND_FIRST,  // Stop after 1 solution (Fastest check)
    STORE_ALL,   // Save and print all boards (RAM heavy, limit N<=16)
    COUNT_ONLY   // Count total solutions (RAM safe, fast for N=17+)
};

class NQueens {
    int n;
    int limit_mask;
    Mode mode;
    std::vector<std::vector<int>> stored_solutions;
    unsigned long long total_count = 0;
    volatile bool found_one = false;

public:
    NQueens(int size, Mode m) : n(size), mode(m) {
        limit_mask = (1 << n) - 1;
    }

    // Returns pair: {count, reference_to_solutions}
    std::pair<unsigned long long, const std::vector<std::vector<int>>&> solve() {
        if (mode == COUNT_ONLY) {
            solve_count();
        } else {
            solve_store();
        }
        return {total_count, stored_solutions};
    }

private:
    // --- OPTIMIZED COUNTING (Low Memory) ---
    void solve_count() {
        #pragma omp parallel reduction(+:total_count)
        {
            #pragma omp for schedule(dynamic)
            for (int col = 0; col < n; ++col) {
                int cols = (1 << col);
                int ld = (1 << col) << 1;
                int rd = (1 << col) >> 1;
                // No vector allocation here, just pure math recursion
                total_count += backtrack_count(1, ld, cols, rd);
            }
        }
    }

    unsigned long long backtrack_count(int row, int ld, int cols, int rd) {
        if (row == n) return 1;

        int possible = ~(ld | cols | rd) & limit_mask;
        unsigned long long local_sum = 0;

        while (possible) {
            int bit = possible & -possible;
            possible -= bit;
            local_sum += backtrack_count(row + 1, (ld | bit) << 1, (cols | bit), (rd | bit) >> 1);
        }
        return local_sum;
    }

    // --- STORING SOLUTIONS (High Memory) ---
    void solve_store() {
        #pragma omp parallel
        {
            std::vector<std::vector<int>> local_solutions;
            std::vector<int> current_path(n);

            #pragma omp for schedule(dynamic)
            for (int col = 0; col < n; ++col) {
                if (mode == FIND_FIRST && found_one) continue;

                int cols = (1 << col);
                int ld = (1 << col) << 1;
                int rd = (1 << col) >> 1;

                current_path[0] = col;
                backtrack_store(1, ld, cols, rd, current_path, local_solutions);
            }

            // Only merge if we actually found something
            if (!local_solutions.empty()) {
                #pragma omp critical
                {
                    stored_solutions.insert(stored_solutions.end(), local_solutions.begin(), local_solutions.end());
                    total_count += local_solutions.size();
                }
            }
        }
    }

    void backtrack_store(int row, int ld, int cols, int rd, 
                         std::vector<int>& path, 
                         std::vector<std::vector<int>>& local_sols) {
        
        if (mode == FIND_FIRST && found_one) return;

        if (row == n) {
            local_sols.push_back(path);
            if (mode == FIND_FIRST) found_one = true;
            return;
        }

        int possible = ~(ld | cols | rd) & limit_mask;

        while (possible) {
            int bit = possible & -possible;
            possible -= bit;
            int col_idx = __builtin_ctz(bit); 
            path[row] = col_idx;

            backtrack_store(row + 1, (ld | bit) << 1, (cols | bit), (rd | bit) >> 1, path, local_sols);
            
            if (mode == FIND_FIRST && found_one) return; 
        }
    }
};

int main(int argc, char* argv[]) {
    // --- 1. SETUP ---
    if (argc < 2) {
        std::cerr << "Usage: ./nqueens_solver <input_file>" << std::endl;
        return 1;
    }

    int n = 0;
    std::ifstream infile(argv[1]);
    if (infile >> n) {
        infile.close();
    } else {
        std::cerr << "Error reading input file." << std::endl;
        return 1;
    }

    // --- 2. SELECT MODE ---
    Mode current_mode;
    if (n <= 16) {
        current_mode = STORE_ALL; // Safe to store in RAM
    } else {
        current_mode = COUNT_ONLY; // N=17+ (Prevents RAM crash)
    }

    std::string modeStr = (current_mode == STORE_ALL) ? "Store All" : "Count Only (High Performance)";
    std::cout << "--- N-Queens Solver (N=" << n << ") ---" << std::endl;
    std::cout << "Mode: " << modeStr << std::endl;

    // --- 3. CALCULATION ---
    auto start_calc = std::chrono::high_resolution_clock::now();

    NQueens solver(n, current_mode);
    auto result = solver.solve();
    unsigned long long count = result.first;
    const auto& solutions_data = result.second;

    auto end_calc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_calc = end_calc - start_calc;

    // --- 4. OUTPUT ---
    std::string output_file = std::string(argv[1]).substr(0, std::string(argv[1]).find_last_of(".")) + "_output.txt";
    auto start_io = std::chrono::high_resolution_clock::now();

    std::ofstream outfile(output_file);
    if (count == 0 && n > 3) { // n=2,3 have 0 solutions
         outfile << "No Solution";
    } else {
        outfile << n << "\n";
        outfile << count << "\n";
        
        // Only write boards if we actually stored them
        if (current_mode == STORE_ALL) {
            for (size_t i = 0; i < solutions_data.size(); ++i) {
                const auto& sol = solutions_data[i];
                for (size_t r = 0; r < sol.size(); ++r) {
                    outfile << (sol[r] + 1) << (r == n - 1 ? "" : " ");
                }
                outfile << "\n";
            }
        }
    }
    outfile.close();

    auto end_io = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_io = end_io - start_io;

    // --- 5. STATS ---
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Solutions Found : " << count << std::endl;
    std::cout << "Calculation Time: " << std::fixed << std::setprecision(6) << duration_calc.count() << " s" << std::endl;
    if (current_mode == COUNT_ONLY) {
         std::cout << "Note: Solutions were counted but NOT written to file (too large)." << std::endl;
    } else {
         std::cout << "File Write Time : " << duration_io.count() << " s" << std::endl;
    }
    std::cout << "-----------------------------------" << std::endl;

    return 0;
}