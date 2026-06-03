// types.h — Common types and numerical utilities for the CMS pipeline
#pragma once

#include <cmath>
#include <string>
#include <variant>

namespace cms {

/// Small epsilon for floating-point division protection (Requirement 14.4)
constexpr double EPSILON = 1e-6;

/// Safe floating-point division that avoids division by zero.
/// Adds EPSILON to the denominator to prevent NaN/Inf propagation.
/// @param numerator   The dividend
/// @param denominator The divisor (will be protected against zero)
/// @return numerator / (denominator + EPSILON)
inline float safeDivide(float numerator, float denominator) {
    return numerator / (denominator + static_cast<float>(EPSILON));
}

/// Double-precision overload of safeDivide.
inline double safeDivide(double numerator, double denominator) {
    return numerator / (denominator + EPSILON);
}

/// Lightweight Result type for C++17 compatibility.
/// Mimics std::expected<T, std::string> semantics (available in C++23).
/// Holds either a success value of type T or an error message string.
template <typename T>
class Result {
public:
    /// Construct a success result
    static Result success(T value) {
        Result r;
        r.data_ = std::move(value);
        return r;
    }

    /// Construct an error result
    static Result error(std::string msg) {
        Result r;
        r.data_ = Error{std::move(msg)};
        return r;
    }

    /// Check if the result holds a value
    bool has_value() const noexcept {
        return std::holds_alternative<T>(data_);
    }

    /// Boolean conversion — true if success
    explicit operator bool() const noexcept { return has_value(); }

    /// Access the value (undefined behavior if error)
    T& value() & { return std::get<T>(data_); }
    const T& value() const& { return std::get<T>(data_); }
    T&& value() && { return std::get<T>(std::move(data_)); }

    /// Access the error message (undefined behavior if success)
    const std::string& error_msg() const& {
        return std::get<Error>(data_).message;
    }

private:
    struct Error {
        std::string message;
    };
    std::variant<T, Error> data_;
};

/// Specialization for void success type (e.g., validation that returns nothing)
template <>
class Result<void> {
public:
    static Result success() {
        Result r;
        r.has_value_ = true;
        return r;
    }

    static Result error(std::string msg) {
        Result r;
        r.has_value_ = false;
        r.error_ = std::move(msg);
        return r;
    }

    bool has_value() const noexcept { return has_value_; }
    explicit operator bool() const noexcept { return has_value_; }

    const std::string& error_msg() const& { return error_; }

private:
    bool has_value_ = false;
    std::string error_;
};

} // namespace cms
