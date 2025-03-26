using LinearAlgebra

function movavg(x, y, n)
    N = length(x) - 1
    A = zeros(N - n + 2, n)     # matrix size is (N-n+2) x n

    for i in 1:n
        A[:, i] = x[i:(N-n+1+i)]
    end
    y_target = Float64.(y[n:end]) # truncate the vector y to fit the dimensions that can be predicted

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

    for n in [1, 2, 3, 5, 10]
        h = movavg(train_x, train_y, n)
        
        y_pred = prediction(test_x, h)

        println("Prediction norm: ", norm(y_pred - test_y[n:end]))
    end
end

main()