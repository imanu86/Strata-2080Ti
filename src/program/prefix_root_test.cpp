#include "strata/program/prefix_root.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static constexpr int32_t turn = 248045;
using Ids = std::vector<int32_t>;
static void check(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}
static int64_t root(const Ids& ids, int64_t min = 2048, bool ssd = true) {
    int64_t last = -1;
    for (int64_t i = int64_t(ids.size()) - 1; i > 0; --i)
        if (ids[size_t(i)] == turn) { last = i; break; }
    return strata::program::prefix_root_boundary(ids, last, turn, min, ssd);
}
static Ids fixture(size_t n, std::initializer_list<size_t> boundaries) {
    Ids ids(n, 7);
    for (size_t i : boundaries) ids.at(i) = turn;
    return ids;
}
static Ids load(const char* path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    check(bool(in), "fixture unavailable");
    auto bytes = in.tellg();
    check(bytes >= 0 && bytes % 4 == 0, "invalid fixture size");
    Ids ids(size_t(bytes) / 4);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(ids.data()), bytes);
    check(bool(in), "fixture read failed");
    return ids;
}
int main(int argc, char** argv) {
    try {
        check(root(fixture(7000, {42, 6000, 6995})) == 6000, "short system must not hide long user root");
        check(root(fixture(7000, {42, 6000, 6995}), 2048, false) == -1, "ordinary RAM policy stops at short system");
        check(root(fixture(8000, {3000, 5000, 7995})) == 3000, "preserve first eligible root");
        check(root(fixture(8000, {3000, 5000, 7995}), 2048, false) == 3000, "ordinary RAM first eligible unchanged");
        check(root(fixture(8000, {2048, 7995})) == 2048, "inclusive minimum");
        check(root(fixture(8000, {2047, 7995})) == -1, "do not select current-turn boundary");
        check(root(fixture(8000, {42, 1000, 7995})) == -1, "no eligible historical boundary");
        check(root(fixture(8000, {})) == -1, "no boundary");
        check(root(fixture(8000, {0, 7999})) == -1, "callsite excludes final current-turn boundary as before");
        auto final_boundary = fixture(8000, {7999});
        check(strata::program::prefix_root_boundary(final_boundary, 8000, turn, 2048, true) == 7999,
              "last ID is a valid lookahead when exclusive bound permits it; no following token needed");
        check(root(fixture(8000, {42})) == -1, "long final user without following turn");
        auto ids = fixture(10000, {42, 3000, 5000, 9995});
        check(root(ids) == 3000, "multiple turns must not overwrite root with later boundary");
        ids[3000] = 7;
        check(root(ids) == 5000, "changed request must recompute root, not reuse stale boundary");
        check(strata::program::prefix_root_boundary(ids, 10001, turn, 2048, true) == -1, "bound outside IDs rejected");
        check(strata::program::prefix_root_boundary(ids, 9995, -1, 2048, true) == -1, "disabled turn token");
        check(strata::program::prefix_root_boundary(ids, 9995, turn, 0, true) == -1, "disabled minimum");
        auto a = fixture(9000, {42, 3000, 8995}), b = a;
        b[7000] = 11;
        auto ra = root(a), rb = root(b);
        check(ra == rb && std::equal(a.begin(), a.begin() + ra + 1, b.begin()), "common prefix includes frontier lookahead");
        b[size_t(rb)] = 8;
        check(!std::equal(a.begin(), a.begin() + ra + 1, b.begin()), "different lookahead cannot be a common cache key");
        if (argc == 1) { std::cout << "PASS: prefix root semantic cases\n"; return 0; }
        check(argc == 3, "usage: prefix_root_test [L01-int32 L02-int32]");
        a = load(argv[1]); b = load(argv[2]);
        check(a.size() == 131248 && b.size() == 131239, "original token counts");
        size_t common = 0;
        while (common < std::min(a.size(), b.size()) && a[common] == b[common]) ++common;
        ra = root(a); rb = root(b);
        check(ra == 113090 && rb == 113090 && common == 131150, "original-ID expected root/common length");
        check(root(a, 2048, false) == -1 && root(b, 2048, false) == -1, "original-ID ordinary RAM root unchanged");
        check(size_t(ra + 1) <= common && a[size_t(ra)] == turn &&
              std::equal(a.begin(), a.begin() + ra + 1, b.begin()), "original root and one-token lookahead common");
        std::cout << "PASS: synthetic semantic cases; L01=131248 L02=131239 common=131150 root=113090 lookahead=248045\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
