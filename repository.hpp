#pragma once
#include <map>
#include <string>
#include <assert.h>
#include "domain.hpp"

struct repository
{

    repository(IJKSize domain, IJKSize halo) : m_domain(domain), m_halo(halo), m_field_size(domain.m_i * domain.m_j * domain.m_k){}

    void make_field(std::string name) {
        Real* ptr = new Real[m_field_size];
        m_fields_h[name] = ptr;
        Real* ptr_d;
        cudaMalloc(&ptr_d, sizeof(Real)*m_field_size);
        m_fields_d[name] = ptr_d;
    }

    Real* field_h(std::string name) {
        assert(m_fields_h[name]);
        return m_fields_h[name];
    }
    Real* field_d(std::string name) {
        assert(m_fields_d[name]);
        return m_fields_d[name];
    }
    void update_host(std::string name) {
        cudaMemcpy(field_h(name), field_d( name), m_field_size * sizeof(Real), cudaMemcpyDeviceToHost);
    }

    void update_device(std::string name) {
        cudaMemcpy(field_d(name), field_h( name), m_field_size * sizeof(Real), cudaMemcpyHostToDevice);
    }

    IJKSize halo() { return m_halo;}
    IJKSize domain() { return m_domain; }
    
private:
    const IJKSize m_domain;
    const IJKSize m_halo;
    const unsigned int m_field_size;
    std::map<std::string, Real*> m_fields_h;
    std::map<std::string, Real*> m_fields_d;

};
