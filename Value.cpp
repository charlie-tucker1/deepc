#include <iostream>


using Scalar = float64_t;


class Value {

public:
    Scalar data;
    Scalar grad;

    Value(Scalar data) : data(data) {
        grad = 0;
    };

private:



public:

    Value operator+(const Value &other) const {
        return this->data + other.data;
    }

    Value operator+(const int other) const {
        return this->data + other;
    }

    Value operator+(const double other) const {
        return this->data + other;
    }

    Value operator-(const Value &other) const {
        return this->data - other.data;
    }

    Value operator-(const int other) const {
        return this->data - other;
    }

    Value operator-(const double other) const {
        return this->data - other;
    }

    Value operator*(const Value &other) const {
        return this->data * other.data;
    }

    Value operator*(const int other) const {
        return this->data * other;
    }

    Value operator*(const double other) const {
        return this->data * other;
    }

    Value operator/(const Value &other) const {
        return this->data / other.data;
    }

    Value operator/(const int other) const {
        return this->data / other;

    }

    Value operator/(const double other) const {
        return this->data / other;
    }


};



int main() {
    return 0;
}