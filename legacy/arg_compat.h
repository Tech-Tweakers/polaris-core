#pragma once
// =============================================================
// Header de compatibilidade temporário — substitui arg.h/console.h/cli.h
// Usado pelo módulo polaris_core para builds recentes do llama.cpp
// =============================================================

#include <algorithm>   // <── adiciona esta
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

// 🧠 Placeholder mínimo para manter compatibilidade
// Define parsing simples de argumentos, usado em exemplos antigos
struct arg_parser {
    std::vector<std::string> args;

    arg_parser() = default;
    arg_parser(int argc, char **argv) {
        for (int i = 0; i < argc; i++) args.emplace_back(argv[i]);
    }

    bool has(const std::string &flag) const {
        return std::find(args.begin(), args.end(), flag) != args.end();
    }

    std::string get(const std::string &flag, const std::string &def = "") const {
        for (size_t i = 0; i + 1 < args.size(); i++) {
            if (args[i] == flag) return args[i + 1];
        }
        return def;
    }
};

// simples log helpers, inspirados no antigo console.h
inline void log_info(const std::string &msg)  { std::cout << "ℹ️  " << msg << std::endl; }
inline void log_warn(const std::string &msg)  { std::cout << "⚠️  " << msg << std::endl; }
inline void log_error(const std::string &msg) { std::cerr << "❌ " << msg << std::endl; }
