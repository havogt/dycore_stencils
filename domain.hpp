#pragma once

struct IJKSize {
    IJKSize(){}
    IJKSize(const unsigned int i, const unsigned int j, const unsigned int k): m_i(i), m_j(j), m_k(k){}
    unsigned int m_i;
    unsigned int m_j;
    unsigned int m_k;
};
