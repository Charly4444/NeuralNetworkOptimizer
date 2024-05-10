from solveretrng import mainRUNNING
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
"""the module solveretrng is an extra from my previous upload on implementation of the method of 
extremepoints; I have used the same exact algorithm to request constraints and return a range

for more see: https://github.com/Charly4444/method_of_extremepoints_optimization.git
"""

# here we will use the range and ANN to search for best solution
if __name__ == "__main__":
    varranges = mainRUNNING()
    # init
    optimal_point = []

    if not varranges: raise(ArithmeticError("NO SOLUTIONS"))
    # make sure we recieved ranges
    else: 
        print(f"NOW WE START WITH THE NETWORK WITH RANGES {varranges}")


        # Step 1: Generate Data
        def generate_data(varranges, num_points):
            generated_points = [np.linspace(start, end, num_points) for start, end in varranges]
            all_points = np.array(np.meshgrid(*generated_points)).T.reshape(-1, len(varranges))
            return all_points

        # Step 2: Approximate Objective Function with ANN
        def train_neural_network(X_train, y_train):
            model = Sequential()
            model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mean_squared_error')
            # Define early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

            # Train the model with early stopping
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

            # # Plot training history (optional)
            # plt.plot(history.history['loss'], label='Training Loss')
            # plt.plot(history.history['val_loss'], label='Validation Loss')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.show()
            
            return model

        # Step 3: Optimization with ANN
        def optimize_with_ann(model, varranges):
            # THIS IS THE OBJECTIVE FUNCTION THAT MY MODEL HAS MIMICKED
            def objective_function(point):
                return model.predict(np.array([point]))[0][0]
            
            # Example of gradient descent optimization
            initial_point = np.mean(varranges, axis=1)  # Start from the midpoint of the varranges
            current_point = initial_point.copy()
            learning_rate = 0.01
            max_iterations = 30
            tolerance = 1e-3

            for iteration in range(max_iterations):
                print(f"iteration {iteration}")
                # Calculate gradient (approximation) using central difference
                grad = np.zeros_like(current_point)
                for i in range(len(current_point)):
                    delta = np.zeros_like(current_point)
                    delta[i] = tolerance
                    grad[i] = (objective_function(current_point + delta) - objective_function(current_point - delta)) / (2 * tolerance)

                # Update current point using gradient descent
                next_point = current_point - learning_rate * grad

                # Clip to stay within bounds defined by the varranges
                # *[list(zip(*varranges))[i] for i in range(len(varranges))] -> complex expression to create clip of boundary
                next_point = np.clip(next_point, *[list(zip(*varranges))[i] for i in range(len(varranges))] )
                
                # Check convergence
                if np.linalg.norm(next_point - current_point) < tolerance:
                    break

                current_point = next_point

            return current_point, objective_function(current_point)



        # ==============================================================
        # Step 1: Generate Data WITH VARRANGES
        # this returns as a numpy array so we can use numpy functions here
        data = generate_data(varranges, num_points=100)

        # Define the objective Function (to TRAIN our MODEL)
        def objective_function(point):
            return 0.06*point[0] + 0.03*point[1] - 0.0004*point[0]*point[1] - 0.02*(point[0]**2) + 0.03*(point[1]**2)

        # Generate target values for the training data
        target_values = np.array([objective_function(point) for point in data]).reshape(-1, 1)

        
        # Step 2: Approximate Objective Function with ANN
        model = train_neural_network(data, target_values)

        # Step 3: Optimization with ANN (PASS THE MODEL AND MAKE OPTIMIZATION)
        optimal_point, min_value = optimize_with_ann(model, varranges)

        print("Optimal point:", optimal_point)
        print("Objective function value at optimal point:", min_value)


# ====================================================
# ===== EXTRAS FOR VISUALIZATIONS ====================
    # x1,x2 = np.meshgrid(np.linspace(*varranges[0],50), np.linspace(*varranges[1],50))

    def generate_pldata(varranges, num_points):
        generated_points = [np.linspace(start, end, num_points) for start, end in varranges]
        all_points = np.array(np.meshgrid(*generated_points))
        return all_points

    def objective_function(data):
        return 0.06 * data[0] + 0.03*data[1] - 0.0004*data[0]*data[1] - 0.02*(data[0]**2) + 0.03*(data[1]**2)

    # Generate data
    # X,y = generate_data([(0, 1), (0, 1)], 100)
    data = generate_pldata(varranges, 50)
    # data[0] :: X
    # data[1] :: y
    
    objective_values = objective_function(data)
    

    # Plot 3D graph
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot()
    ax.plot_surface(data[0], data[1], objective_values, cmap='viridis', alpha=0.8)

    # Plot solution point
    ax.scatter(optimal_point[0], optimal_point[1], objective_function(optimal_point), color='r', marker='o', s=30, label='optimum in range')

    # Plot dashed line
    ax.plot([optimal_point[0], optimal_point[0]], [optimal_point[1], optimal_point[1]], [0, objective_function(optimal_point)], color='k', linestyle='--')

    # plot constraints
    myX = np.linspace(*varranges[0],15); myY = np.linspace(*varranges[1],15)
    # constraint1
    ax.plot(myX, 1-myX, linestyle='--', label='cons1')
    # constraint2
    ax.plot(myX, (4-2*myX)/3, linestyle='--', label='cons2')
    # constraint3
    ax.plot(myX, (2-3*myX)/2, linestyle='--', label='cons3')
    # constraint4
    ax.plot(0.8*np.ones_like(myX), myY, linestyle='--', label='cons4')
    # constraint5
    ax.plot(myX, 0.3*np.ones_like(myY), linestyle='--', label='cons5')
    # # optimal
    # ax.scatter(optimal_point[0], optimal_point[1], color='r', marker='o', s=30, label='optimum in range')

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Objective Function, f(x1,x2)")
    # ax.set_title("boundary lines")
    # plt.legend()
    plt.show()
