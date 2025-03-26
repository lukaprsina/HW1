using LinearAlgebra
using Printf
using Plots
using Statistics

function movavg(x, y, n)
    N = length(x) - 1
    A = zeros(N - n + 2, n)     # matrix size is (N-n+2) x n

    for i in 1:n
        A[:, i] = x[i:(N-n+1+i)]
    end

    println("n = $n, cond(A) = $(cond(A))")

    y_target = Float64.(y[n:end]) # truncate the vector y to fit the dimensions that can be predicted

    # Aplus = pinv(A)
    # return Aplus * y_target'
    return A \ y_target
end

function prediction(x::Vector, h::Vector)
    N = length(x) - 1
    n = length(h)
    m = length(x)
    @assert m >= n "Input x must have at least as many elements as the coefficients h"

    A = zeros(N - n + 2, n)

    for i in 1:n
        A[:, i] = x[i:N-n+i+1]
    end

    return A * h
end

function main()
    train_lines = readlines("io-train.txt")
    train_x = parse.(Float64, split(train_lines[2]))
    train_y = parse.(Float64, split(train_lines[3]))

    test_lines = readlines("io-test.txt")
    test_x = parse.(Float64, split(test_lines[2]))
    test_y = parse.(Float64, split(test_lines[3]))

    println("n\tTrain Norm\tTrain MSE\tTest Norm\tTest MSE")
    for n in [1, 2, 3, 5, 10]
        h = movavg(train_x, train_y, n)
        
        # Training set predictions and errors
        y_train_pred = prediction(train_x, h)
        train_y_valid = train_y[n:end]  # Align with predictable indices
        train_error = y_train_pred - train_y_valid
        train_norm = norm(train_error)
        train_mse = mean(train_error.^2)  # MSE = average squared error

        # Test set predictions and errors
        y_test_pred = prediction(test_x, h)
        test_y_valid = test_y[n:end]
        test_error = y_test_pred - test_y_valid
        test_norm = norm(test_error)
        test_mse = mean(test_error.^2)

        # Formatted output
        Printf.@printf("%d\t%.2f\t\t%.4f\t\t%.2f\t\t%.4f\n",
                n, train_norm, train_mse, test_norm, test_mse)
    end

    h_n5 = movavg(train_x, train_y, 5)
    h_n10 = movavg(train_x, train_y, 10)
    plot(h_n5, label = "n = 5", title = "Coefficient Magnitudes")
    plot!(h_n10, label = "n = 10")
end

main()
