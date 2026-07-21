struct Position {
    int x;
    int y;

    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Position& other) const {
        return !(*this == other);
    }

    bool operator<(const Position& other) const {
        if (x != other.x) {
            return x < other.x;
        }
        return y < other.y;
    }

    bool operator<=(const Position& other) const {
        return *this < other || *this == other;
    }

    bool operator>(const Position& other) const {
        return !(*this <= other);
    }

    bool operator>=(const Position& other) const {
        return !(*this < other);
    }
};