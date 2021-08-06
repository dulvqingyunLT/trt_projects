#ifndef TRT_UTILS_H
#define TRT_UTILS_H

#include<map>
#include "NvInfer.h"

namespace LGT{
namespace spanner{

std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names, int max_image_num=512) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr && file_names.size() < max_image_num) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            // std::string cur_file_name(p_dir_name);
            // cur_file_name += "/";
            // cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

// static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
//     DIR *p_dir = opendir(p_dir_name);
//     if (p_dir == nullptr) {
//         return -1;
//     }

//     struct dirent* p_file = nullptr;
//     while ((p_file = readdir(p_dir)) != nullptr) {
//         if (strcmp(p_file->d_name, ".") != 0 &&
//             strcmp(p_file->d_name, "..") != 0) {
//             // std::string cur_file_name(p_dir_name);
//             // cur_file_name += "/";
//             // cur_file_name += p_file->d_name;
//             std::string cur_file_name(p_file->d_name);
//             file_names.push_back(cur_file_name);
//         }
//     }

//     closedir(p_dir);
//     return 0;
// }

}  //namespace spanner  
} //namespace LGT

#endif