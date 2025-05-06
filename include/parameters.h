#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <typeinfo>
#include <unordered_map>
#include <boost/json.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace numaann
{
    class Parameters
    {
    public:
        template <typename ParamType>
        inline void Set(const std::string &name, const ParamType &value)
        {
            std::stringstream sstream;
            sstream << value;
            params[name] = sstream.str();
        }

        template <typename ParamType>
        inline ParamType Get(const std::string &name) const
        {
            auto item = params.find(name);
            if (item == params.end())
            {
                throw std::invalid_argument("Invalid parameter name.");
            }
            else
            {
                return ConvertStrToValue<ParamType>(item->second);
            }
        }

        template <typename ParamType>
        inline ParamType Get(const std::string &name, const ParamType &default_value)
        {
            try
            {
                return Get<ParamType>(name);
            }
            catch (std::invalid_argument e)
            {
                return default_value;
            }
        }

        void LoadConfigFromJSON(const std::string &config_file_path)
        {
            std::cout << "config_file_path: " << config_file_path << std::endl;
            std::ifstream inputFile(config_file_path);
            if (!inputFile.is_open())
            {
                throw std::runtime_error("error@LoadConfigFromJSON: cant't find " + config_file_path);
            }

            std::string jsonString((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());

            boost::system::error_code ec;
            boost::json::value jsonValue = boost::json::parse(jsonString, ec);
            if (ec)
                throw std::runtime_error("error@LoadConfigFromJSON: 解析JSON文件失败 " + ec.message());

            try
            {
                boost::json::object jsonObject = jsonValue.as_object();
                // this->base_file_path = jsonObject["base_file_path"].as_string().c_str();
                for (auto iter = jsonObject.begin(); iter != jsonObject.end(); iter++)
                {
                    std::string key(iter->key_c_str());
                    std::string value(iter->value().as_string().c_str());
                    std::cout << "Load Config...... " << key << ": " << value << std::endl;
                    Set(key, value);
                }
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("error@LoadConfigFromJSON: 访问JSON数据失败 " + std::string(e.what()));
            }
        }

    private:
        std::unordered_map<std::string, std::string> params;

        template <typename ParamType>
        inline ParamType ConvertStrToValue(const std::string &str) const
        {
            std::stringstream sstream(str);
            ParamType value;
            if (!(sstream >> value) || !sstream.eof())
            {
                std::stringstream err;
                err << "Failed to convert value '" << str << "' to type: " << typeid(value).name();
                throw std::runtime_error(err.str());
            }
            return value;
        }
    };

}
