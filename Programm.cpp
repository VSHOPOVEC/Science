#include "INCLUDE.h"

float ToFindNumberFromString(const std::string& string) {
    auto stream = std::istringstream(string);
    float num;
    while (stream >> num) {
        if (num != 0) {
            return num;
        }
    }
    return 0;
}

std::vector <std::vector <float>> ToDownloadArray(std::string name_of_txt_file) {
    std::ifstream file_for_reading(name_of_txt_file + ".txt");
    std::vector <std::vector<float>> result;
    if (!file_for_reading.is_open()) {
        std::cout << "Error\n";
    }
    else {
        std::cout << "File " + name_of_txt_file + " is open!\n";
        std::string temp;
        std::vector<float> position;
        while (std::getline(file_for_reading, temp)) {
            if (temp.find("(mm,mm,mm,deg):") != std::string::npos) {
                position.push_back(ToFindNumberFromString(temp.substr(15, temp.length())));
            }
            else if (temp.find("[DATA BLOCK") != std::string::npos) {
                std::vector <float> temp_vector;
                while (std::getline(file_for_reading, temp) && !temp.empty()) {
                    std::istringstream stringstream(temp);
                    float number, speed, voltage;
                    stringstream >> number >> speed >> voltage;
                    temp_vector.push_back(speed);
                }
                result.push_back(temp_vector);
            }
        }
        result.push_back(position);
    }
    return result;
}

std::vector <float> Unpicking(std::vector<std::vector<float>>& vector) {
    std::vector<float> end = vector.back();
    vector.pop_back();
    return end;
}

std::vector <float> Result(const std::vector < std::vector < float >>& vector) {
    std::vector <float> result;
    float float_result = 0;
    for (auto pointer1 : vector) {
        for (auto pointer2 : pointer1) {
            float_result += pointer2;
        }
        float_result = float_result / (pointer1.size());
        result.push_back(float_result);
    }
    return result;
}


std::vector <float> StandardDeviation(const std::vector<std::vector<float>>& vector) {
    std::vector <float> result;
    for (auto pointer : vector) {
        float sum = 0; 
        float sq_sum = 0; 
        for (auto pointer2 : pointer) {
            sum += pointer2;
        }
        float middle = sum / (pointer.size()); 
        for (auto pointer1 : pointer) {
            sq_sum += pow(middle - pointer1, 2); 
        }
        float temp_result = sq_sum / (pointer.size());
        result.push_back(pow(temp_result, 0.5)); 
    }
    return result; 
}

void To_Print_Array(std::vector<float>& vector) {
    for (auto p1 : vector) {
        std::cout << p1 << " ";
    }
}

void ToWriteToTXT(const std::vector <float>& vector, const std::vector <float>& position, const std::vector <float> & StandardDeviation, std::string name_of_file) {
    std::ofstream stream;
    stream.open("result_of_" + name_of_file + ".txt");
    stream << "result_of_" + name_of_file + " = " + "[";
    for (int i = 0; i < vector.size() - 1; ++i) {
        stream << vector[i] << ",";
    }
    stream << vector[vector.size() - 1];
    stream << "]";
    stream << std::endl;
    stream << "position_of_" + name_of_file + " = [";
    for (int i = 0; i < position.size() - 1; ++i) {
        stream << position[i] << ",";
    }
    stream << position[position.size() - 1]; 
    stream << "]";
    stream << std::endl;

    stream << "StandardDeviation_of_" + name_of_file + " = " + "[";
    for (int i = 0; i < StandardDeviation.size() - 1; ++i) {
        stream << StandardDeviation[i] << ",";
    }
    stream << StandardDeviation[StandardDeviation.size() - 1];
    stream << "]";
    stream << std::endl;
}


void Programm(std::string name_of_file) {
    auto result = ToDownloadArray(name_of_file);
    auto position = Unpicking(result);
    auto result2 = Result(result);
    auto StandardDeviation_ = StandardDeviation(result);
    ToWriteToTXT(result2, position, StandardDeviation_, name_of_file);
}