import numpy as np
import standard_form as sfc
from itertools import combinations
"""
This program will only solve problems where all variables 
x1, x2, ... , xn >= 0
if theres one or more varibles that dont satisfy this, then
use the second solver.
"""

def mainRUNNING():

    solutionSet = []

    data = sfc.allRECOMENDED()
    # data format recieved {'num_orig_vars': '', 'A': [[]], 'b': [], 'C': []}
    num_orig_vars = data['num_orig_vars']
    A_matrix = data['A']
    b_matrix = data['b']
    C_matrix = data['C']

    # function to generate all square matrices
    def generate_square_matrices(A_mx, C_mx):
        all_setparty = []
        m, n = A_mx.shape
        # get all combs
        index_tuples = list(combinations(range(n), m))

        for index in index_tuples:
            onepart = {}
            onepart['A_mx'] = A_mx[:, index]
            onepart['C_mx'] = C_mx[list(index)]     #b/c => C_mx is a 1D array
            onepart['index'] = index
            
            all_setparty.append(onepart)
        return all_setparty



    # ====================compute Rank of matrix A ==================
    rank = np.linalg.matrix_rank(A_matrix)
    print(f"Rank of the matrix: {rank}\n")
    numrows = A_matrix.shape[0]

    if numrows == rank:
        ## if its a full-rank matrix, generate all possible, m-by-m matrices

        ## {A_mx:[[ndarray]], C_mx:[[ndarray]], index: tuple}
        sets_square_matrices = generate_square_matrices(A_matrix, C_matrix)
        
        ## for-each matrix check if determinant is not 0, Proceed
        for sq_mtrx in sets_square_matrices:
            F = sq_mtrx['A_mx']
            C = sq_mtrx['C_mx']

            det = np.linalg.det(F)
            print(f"{F}, det:= {det}\n")

            if det != 0:
                ## so, its invertible we solve and get value

                ## Solve the linear system Fx = b_matrix
                x = np.linalg.solve(F, b_matrix)
                print(f"Solution vector x: {x}\n")
                
                ## then, store as possible soln
                ## I am converting to list because I will use JSON  
                if np.all(x > 0):
                    sol = {}
                    sol['x'] = x.tolist()
                    sol['index'] = sq_mtrx['index']
                    sol['C'] = C.tolist()
                    solutionSet.append(sol)
                
            else:
                print('This matrix is not invertible!\n')
    else:
        print('you entered an incompatible matrix, start again!!\n')


    # ==================================================================================
        # LETS FETCH THE CHOSEN SOLUTION

    print(f'the solution set: {solutionSet}')
    if solutionSet:
        # basically evaluatingthe objectives (but i wont do that)
        # feasible_solutions = [sol for sol in solutionSet if all(idx < num_orig_vars for idx in sol['index'])]
        
        # special feasible solution chosen now such that at least any of the original variables exist
        feasible_solutions = [sol for sol in solutionSet if any(idx < num_orig_vars for idx in sol['index'])]

        # We choose the feasible solution (ACTIVATE TO STREAMLINE FURTHER)
        feasible_results = [np.dot(sol['x'], sol['C']) for sol in feasible_solutions]
        min_index_feasible = np.argmin(feasible_results)
        chosen_feasible_solution = solutionSet[min_index_feasible]

        # or for now we just pick the first guy in the set of feasible and search
        # assuming as based on this method that at least a solution was recovered
        chosen_feasible_solution = feasible_solutions[0]
        
        # print feasible solution only (the feasible solution we picked)
        for i, idx in enumerate(chosen_feasible_solution['index']):
            print(f"feasible_sol: x{idx} = {chosen_feasible_solution['x'][i]}")

        # =============================================================
        # NOW WERE FITTING IN THE RECREATION OF RANGES

        # Initialize var_ranges list
        var_ranges = []

        # Iterate through the range of variables
        for i in range(num_orig_vars):
            if i in chosen_feasible_solution['index']:
                # If variable exists in the solution, get its range
                var_index = chosen_feasible_solution['index'].index(i)
                var_value = chosen_feasible_solution['x'][var_index]
                var_ranges.append((round(var_value - var_value,2), round(var_value + var_value,2)))
            else:
                # If variable doesn't exist in the solution, set its range to (0, 0)
                var_ranges.append((0, 0))

        print("var_ranges: ",var_ranges)

    else:
        print("No solutions found!!")

    print("chosen_feasible_solution: ",chosen_feasible_solution)


    # HAND CONTROL TO THE MAINNW FUNCTION
    return var_ranges
