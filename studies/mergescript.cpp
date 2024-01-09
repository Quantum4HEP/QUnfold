#include <iostream>
#include <TFile.h>
#include <TFileMerger.h>
#include <TTree.h>
#include <dirent.h>
#include <cstdio>

std::string list_recovars[12] = {
    "Electron_size",
    "Muon_size",
    "Electron.PT",
    "Muon.PT",
    "Electron.Eta",
    "Muon.Eta",
    "Electron.Phi",
    "Muon.Phi",
    "Jet.Flavor",
    "Jet.PT",
    "Jet.Eta",
    "Jet.Phi",
};
std::string list_partvars[5] = {"Particle.PT",
    "Particle.Eta",
    "Particle.Phi",
    "Particle.E",
    "Particle.PID"};

void skimmer(std::string inputfile, bool single_file_flag = true){

    TFile* f1 = new TFile(inputfile.c_str());
    std::string treename = f1->GetListOfKeys()->Last()->GetName();


    TTree *oldtree;
    f1->GetObject(treename.c_str(), oldtree);

    // Deactivate all branches
    oldtree->SetBranchStatus("*", 0);
    // Activate only few of them
    
    if (treename == "LHEF"){
    for (auto activeBranchName : list_partvars)
       oldtree->SetBranchStatus(activeBranchName.c_str(), 1);}
    else if (treename == "Delphes"){
        for (auto activeBranchName : list_recovars)
       oldtree->SetBranchStatus(activeBranchName.c_str(), 1);
    }
    else{
        std::cout << "Tree name not supported, check TFile\n";
    }
    // Create a new file + a clone of old tree in new file
    
    
    if(single_file_flag){
    std::string newtreename = inputfile.substr(0, inputfile.length() - 5) + "_skimmed.root";
    TFile newfile(newtreename.c_str(), "recreate");
    auto newtree = oldtree->CloneTree();
    newtree->Print();
    newfile.Write();
    }
    else{
    TFile newfile("temp.root", "recreate");
    auto newtree = oldtree->CloneTree();
    newtree->Print();
    newfile.Write();
    }
    

}
int merger(std::string outname,std::string rundirs_path = "Events") {

    std::cout << "Merging " << outname << std::endl;

    std::cout << "Max tree size: " << TTree::GetMaxTreeSize() << std::endl;
    TTree::SetMaxTreeSize(600000000000); // 600 Gb
    std::cout << "Updated tree size: " << TTree::GetMaxTreeSize() << std::endl;

    TFileMerger rm(false);
    rm.SetFastMethod(true);

    std::string path = rundirs_path;
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
        skimmer((F + std::string("/unweighted_events.root")).c_str(),false); //tag_1_delphes_events.root
        rm.AddFile("temp.root");
    }

    rm.OutputFile(file_output.c_str());
    rm.Merge();
    std::remove("temp.root");
    return 0;
}



            