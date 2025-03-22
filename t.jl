

function prediction(x, h)
    n = length(h)
    m = length(x) - n + 1
    m >= 1 || throw(ArgumentError("Input x must have length ≥ $(n)"))
    y_pred = zeros(eltype(x), m)
    for k in 1:m
        y_pred[k] = dot(h, x[k:(k+n-1)])
    end
    return y_pred
end

function prediction2(x::Vector, h::Vector)
    n = length(h)
    N = length(x) - 1

    # The prediction is valid for t = n - 1 to N
    start_time = n - 1
    end_time = N

    num_predictions = end_time - start_time + 1
    if num_predictions <= 0
        return Float64# Return empty vector if no predictions can be made
    end

    y_predicted = zeros(num_predictions)

    for i in 1:num_predictions
        t = start_time + i - 1 # Current time index (0-based)
        predicted_value = 0.0
        for j in 1:n
            # Index of x to use: t - n + j
            x_index = t - n + j + 1 # Adjust for Julia's 1-based indexing
            predicted_value += h[j] * x[x_index]
        end
        y_predicted[i] = predicted_value
    end

    return y_predicted
end
