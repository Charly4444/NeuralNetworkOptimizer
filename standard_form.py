import numpy as np
"""
This program is written to take any set of Linear programming expressions
and, return the correspoinding A and b matrices.
The coefficients of the expressions must be entered in the same order,
User doesnt need to pay attention to the slack coefficients, it is 
automatically handled in the program
"""

# A_matrix = []
# b_matrix = []
# C_Matrix = []


# read in all constraints
def get_constraint(num_orig_vars,num_constraints,all_constraints):
    all_myconstraints = []

    total_vars = num_orig_vars

    # get constraint
    constraintsArray = []
    for i in range(num_constraints):
        constraint = input(f"Enter the {i+1}th constraint : ").split()
        constraintsArray.append(constraint)
    
    # GOING THROUGH TO BUILD THE CONSTRAINTS
    for constraint_parts in constraintsArray:

        constraint = {}
        
        constraint['constraint_coef'] = constraint_parts[:-2]
        constraint['constraint_op'] = constraint_parts[-2]
        constraint['constraint_cons'] = constraint_parts[-1]
        
        # update no of vars
        if constraint_parts[-2] == '<=':
            constraint['slack'] = 1
            total_vars+=1
        elif constraint_parts[-2] == '>=':
            constraint['slack'] = -1
            total_vars+=1
        else:
            constraint['slack'] = 0
            
        all_myconstraints.append(constraint)
    all_constraints[:] = all_myconstraints
    return total_vars

# build A matrix
def build_Amatrix(num_orig_vars, final_num_vars,all_constraints):
    A_matrixofChars = []

    i = 0
    for cons in all_constraints:
        if cons['slack']==1 or cons['slack']==-1:
            if i!=0:
                num_zeroes_before = i*[0]
            else:
                num_zeroes_before=''
            sk=[cons['slack']]
            if ((final_num_vars - num_orig_vars -1 - i) != 0):
                num_zeroes_after = (final_num_vars - num_orig_vars - 1 - i)*[0]
            else:
                num_zeroes_after=''
            i+=1  

        else:
            num_zeroes_before=''
            sk=''
            if ((final_num_vars - num_orig_vars) !=0):
                num_zeroes_after = (final_num_vars - num_orig_vars)*[0]
            else:
                num_zeroes_after=''
        
        # concatenate and append to A_matrixofChars => that completes the basic cycle
        A_matrixofChars.append(cons['constraint_coef'] + list(map(str, num_zeroes_before)) + list(map(str, sk)) + list(map(str, num_zeroes_after)))
    # print(f"row: {A_matrixofChars}\n")
    
    arrays_of_numbers = [np.array(list(map(float, sublist))) for sublist in A_matrixofChars]
    stacked_Amatrix = np.vstack(arrays_of_numbers)
    return stacked_Amatrix


def build_Bmatrix(all_constraints):
    b_values = [float(cons['constraint_cons']) for cons in all_constraints]
    b_matrix = np.array(b_values)
    return b_matrix


# # get terms of the objectuive function
# def get_objectivefcn(num_orig_vars):
#     objective_part = input(f"Enter the {num_orig_vars} variables of objectiveFunction as 'a1 a2 ... an: ").split()
#     return objective_part

def build_Cmatrix(num_orig_vars, final_num_vars):
    # take objective
    objective_part = input(f"Enter the {num_orig_vars} variables of objectiveFunction as 'a1 a2 ... an: ").split()
    
    if ((final_num_vars - num_orig_vars) !=0):
        num_zeroes_after = (final_num_vars - num_orig_vars)*[0]
    else:
        num_zeroes_after=''
    
    c_valuesChar = objective_part + list(map(str, num_zeroes_after))
#   = = = = =
    c_values = [float(val) for val in c_valuesChar]
    c_matrix = np.array(c_values)
    return c_matrix










# ALL THE MAIN PART OF THE FUNCTION
def allRECOMENDED():
    # some initial vars
    all_constraints = []

    # no of vars
    num_orig_vars = int(input("Enter the number of original decision variables: "))
    
    # no of constraints
    num_constraints = int(input("Enter the number of constraints: "))
    
    # get constraint
    final_num_vars = get_constraint(num_orig_vars,num_constraints,all_constraints)
    

    # build A matrix
    A_matrix = build_Amatrix(num_orig_vars, final_num_vars,all_constraints)
    print(f"A: {A_matrix}\n")

    # build b_Matrix
    b_matrix = build_Bmatrix(all_constraints)
    print(f"B: {b_matrix}\n")

    # build C_Matrix / <<this includes getting the objective function>>
    c_matrix = build_Cmatrix(num_orig_vars, final_num_vars)
    print(f"C: {c_matrix}")


# ========================== AESTHETIC CODE JUST TO SHOW THE ELEMENTS=========================
    # def print_matrix(matrix):
    #     for row in matrix:
    #         row_str = ', '.join(map(str, row))
    #         print(row_str)

    # print_matrix(A_matrix)
# ============================================================================================
    return {'num_orig_vars': num_orig_vars, 'A': A_matrix, 'b': b_matrix, 'C': c_matrix}
