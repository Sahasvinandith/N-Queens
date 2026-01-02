#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <iomanip> // For nice formatting of time

// Global flag to stop threads early if we only need 1 solution
volatile bool found_one = false;

struct Solution {
    std::vector<int> board; // Stores column index for each row
};

class NQueens {
    int n;
    int limit_mask;
    bool find_all;
    std::vector<Solution> solutions;

public:
    NQueens(int size, bool all) : n(size), find_all(all) {
        limit_mask = (1 << n) - 1;
    }

    std::vector<Solution> solve() {
        #pragma omp parallel
        {
            std::vector<Solution> local_solutions;
            std::vector<int> current_path(n);

            #pragma omp for schedule(dynamic)
            for (int col = 0; col < n; ++col) {
                if (!find_all && found_one) continue;

                int cols = (1 << col);
                int ld = (1 << col) << 1;
                int rd = (1 << col) >> 1;

                current_path[0] = col;
                backtrack(1, ld, cols, rd, current_path, local_solutions);
            }

            #pragma omp critical
            {
                solutions.insert(solutions.end(), local_solutions.begin(), local_solutions.end());
            }
        }
        return solutions;
    }

private:
    void backtrack(int row, int ld, int cols, int rd, 
                   std::vector<int>& path, 
                   std::vector<Solution>& local_solutions) {
        
        if (!find_all && found_one) return;

        if (row == n) {
            local_solutions.push_back({path});
            if (!find_all) found_one = true;
            return;
        }

        int possible = ~(ld | cols | rd) & limit_mask;

        while (possible) {
            int bit = possible & -possible;
            possible -= bit;
            int col_idx = __builtin_ctz(bit); 
            path[row] = col_idx;

            backtrack(row + 1, (ld | bit) << 1, (cols | bit), (rd | bit) >> 1, path, local_solutions);
            
            if (!find_all && found_one) return; 
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./nqueens_solver <input_file>" << std::endl;
        return 1;
    }
    std::string input_file = argv[1];
    std::string output_file = input_file.substr(0, input_file.find_last_of(".")) + "_output.txt";

    int n = 0;
    std::ifstream infile(input_file);
    if (infile >> n) {
        infile.close();
    } else {
        std::cerr << "Error reading input file." << std::endl;
        return 1;
    }

    // --- STRATEGY SELECTION ---
    bool find_all = (n < 16); 

    std::cout << "--- N-Queens Solver (N=" << n << ") ---" << std::endl;
    std::cout << "Mode: " << (find_all ? "Find ALL Solutions" : "Find First Solution") << std::endl;

    // --- TIMER START: CALCULATION ---
    auto start_calc = std::chrono::high_resolution_clock::now();

    NQueens solver(n, find_all);
    auto solutions = solver.solve();

    // --- TIMER STOP: CALCULATION ---
    auto end_calc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_calc = end_calc - start_calc;


    // --- TIMER START: I/O WRITE ---
    auto start_io = std::chrono::high_resolution_clock::now();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Solutions Found : " << solutions.size() << std::endl;
    std::cout << "Calculation Time: " << duration_calc.count() << " seconds" << std::endl;
    
    std::ofstream outfile(output_file);
    if (solutions.empty()) {
        outfile << "No Solution";
    } else {
        outfile << n << "\n";
        outfile << solutions.size() << "\n";
        
        for (size_t i = 0; i < solutions.size(); ++i) {
            const auto& sol = solutions[i];
            for (size_t r = 0; r < sol.board.size(); ++r) {
                outfile << (sol.board[r] + 1) << (r == n - 1 ? "" : " ");
            }
            outfile << "\n";
            if (find_all && i < solutions.size() - 1) outfile << "\n";
        }
    }
    outfile.close();

    // --- TIMER STOP: I/O WRITE ---
    auto end_io = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_io = end_io - start_io;


    // --- PRINT STATS TO CONSOLE ---
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Solutions Found : " << solutions.size() << std::endl;
    std::cout << "Calculation Time: " << duration_calc.count() << " seconds" << std::endl;
    std::cout << "File Write Time : " << duration_io.count() << " seconds" << std::endl;
    std::cout << "Total Time      : " << (duration_calc.count() + duration_io.count()) << " seconds" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    return 0;
}