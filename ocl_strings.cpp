#include "ocl_strings.hpp"

#include <algorithm>
#include <sstream>
#include <iostream>

std::string matchParam
(std::ifstream& input, const char* param_name)
{
    std::string param_string;
    std::string line;

    /* read in line from file */
    while (std::getline(input, line))
    {
        if (line.find("!") != std::string::npos) continue;
        /* if it has the parameter name, its the line we want */
        if (line.find(param_name) != std::string::npos)
        {
            if (line.find("=") != std::string::npos)
            {
                param_string = std::string(line.erase(0, 1+line.find("=")));
            }
            else
            {
                param_string = std::string(param_name);
            }
            break;
        }
    }

    // getline() sets failbit - clear it
    input.clear();
    input.seekg(0);

    return param_string;
}

std::string readString
(std::ifstream& input, const char * setting)
{
    std::string plat_name = matchParam(input, setting);

    // convert to lower case
    std::transform(plat_name.begin(),
                   plat_name.end(),
                   plat_name.begin(),
                   tolower);

    return plat_name;
}

int readInt
(std::ifstream& input, const char * setting)
{
    std::string param_string = matchParam(input, setting);

    int param_value;

    if (param_string.size() == 0)
    {
        // not found in file
        param_value = -1;
    }
    else
    {
        std::stringstream converter(param_string);

        if (!(converter >> param_value))
        {
            param_value = -1;
        }
    }

    return param_value;
}


int typeMatch
(std::string& type_name)
{
    //fprintf(stderr, "Matching with %s\n", type_name.c_str());

    // match
    if (type_name.find("cpu") != std::string::npos)
    {
        return CL_DEVICE_TYPE_CPU;
    }
    else if (type_name.find("gpu") != std::string::npos)
    {
        return CL_DEVICE_TYPE_GPU;
    }
    else if (type_name.find("accelerator") != std::string::npos)
    {
        return CL_DEVICE_TYPE_ACCELERATOR;
    }
    else if (type_name.find("all") != std::string::npos)
    {
        return CL_DEVICE_TYPE_ALL;
    }
    else if (type_name.size() == 0)
    {
        return CL_DEVICE_TYPE_ALL;
    }
    else
    {
        return 0;
    }
}

std::string strType
(cl_device_type dtype)
{
    switch (dtype)
    {
    case CL_DEVICE_TYPE_GPU :
        return std::string("GPU");
    case CL_DEVICE_TYPE_CPU :
        return std::string("CPU");
    case CL_DEVICE_TYPE_ACCELERATOR :
        return std::string("ACCELERATOR");
    default :
        return std::string("Device type does not match known values");
    }
}

bool paramEnabled
(std::ifstream& input, const char* param)
{
    std::string param_string = matchParam(input, param);
    return (param_string.size() != 0);
}

