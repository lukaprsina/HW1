using LinearAlgebra

function movavg(x::Vector, y::Vector, n::Int)
    N = length(x) - 1  # Since x has N+1 elements (t=0 to N)
    # Number of equations: N - n + 2
    rows = N - n + 2
    # Initialize matrix A
    A = zeros(rows, n)
    # Initialize vector y_target (extract the corresponding y values)
    y_target = zeros(rows)

    for (row_idx, t) in enumerate(n-1:N)  # t ranges from n-1 to N
        # Fill each row of A with x(t-n+1) to x(t)
        A[row_idx, :] = x[t-n+2:t+1]  # +1 for 1-based indexing
        y_target[row_idx] = y[t+1]  # y[t] is at y[t+1] in 1-based
    end

    # Solve the least squares problem Ah ≈ y_target
    h = A \ y_target
    return h
end

function movavg_erik(x, y, n)
    N = length(x) - 1
    A = zeros(N - n + 2, n)     # matrix size is (N-n+2) x n

    # make matrix A by columns:
    for i in 1:n
        A[:, i] = x[i:N-n+1+i]
    end

    y = Float64.(y[1:N-n+2])' # truncate the vector y to fit the dimensions that can be predicted

    Aplus = pinv(A)
    return Aplus * y'
end

function movavg_erik2(x, y, n)
    N = length(x) - 1
    A = zeros(N - n + 2, n)     # matrix size is (N-n+2) x n

    # make matrix A by columns:
    for i in 1:n
        A[:, i] = x[i:(N-n+1+i)]
    end

    # POPRAVEK ENA: ni treba transponirati y
    y_target = Float64.(y[n:end]) # truncate the vector y to fit the dimensions that can be predicted

    # POPRAVEK DVA: pinv je manj učinkovit za sparse matrix
    # Aplus = pinv(A)
    # return Aplus * y'

    return A \ y_target
end

function prediction(x::Vector, h::Vector)
    n = length(h)
    m = length(x)
    @assert m >= n "Input x must have at least as many elements as the coefficients h"
    y_pred = zeros(eltype(x), m - n + 1)  # Initialize output vector
    for i in 1:(m-n+1)
        # Extract the window of size n starting at index i
        window = @view x[i:(i+n-1)]
        # Compute the dot product of h and the window
        y_pred[i] = dot(h, window)
    end
    return y_pred
end

function prediction_erik(x::Vector, h::Vector)
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
        h = movavg3(train_x, train_y, n)
        h_erik = movavg_erik(train_x, train_y, n)
        h_erik2 = movavg_erik2(train_x, train_y, n)
        y_pred = prediction(test_x, h)
        y_pred_erik = prediction_erik(test_x, h_erik)
        y_pred_erik2 = prediction_erik(test_x, h_erik2)
        println("n = ", n, ", razlika: ", (h - h_erik2))
        # println("n = ", n)
        # println("Coefficients: ", h)
        # Use test_y[n:end] to align dimensions with y_pred.
        # println("Prediction norm: ", norm(y_pred - test_y[n:end]))
        # println("Prediction norm: ", norm(y_pred - test_y[n:end]), ", erik: ", norm(y_pred_erik - test_y[n:end]))
    end
end

main()