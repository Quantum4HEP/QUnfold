#include <iostream>
#include <TFile.h>
#include <TFileMerger.h>
#include <TTree.h>
#include <dirent.h>

int merger(std::string outname) {

    std::cout << "Merging " << outname << std::endl;

    std::cout << "Max tree size: " << TTree::GetMaxTreeSize() << std::endl;
    TTree::SetMaxTreeSize(600000000000); // 600 Gb
    std::cout << "Updated tree size: " << TTree::GetMaxTreeSize() << std::endl;

    TFileMerger rm(false);
    rm.SetFastMethod(true);

    std::string path = "Events";
    std::string file_output = outname + std::string(".root");
    std::vector<std::string> file_list;
    DIR* dir = opendir(path.c_str());

    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_DIR && entry->d_name[0] != '.') {
                file_list.push_back(path + std::string("/") + std::string(entry->d_name));
            }
        }

        closedir(dir);
    } else {
        std::cerr << "Error opening directory." << std::endl;
    }


    std::cout << "Input file list:";
    for (const auto& file : file_list) {
        std::cout << " " << file;
    }
    std::cout << std::endl;

    std::cout << "Output file: " << file_output << std::endl;

    for (const auto& F : file_list) {
        std::cout << "Adding -> " << F << std::endl;
        rm.AddFile((F + std::string("/tag_1_delphes_events.root")).c_str());
    }

    rm.OutputFile(file_output.c_str());
    rm.Merge();

    return 0;
}
