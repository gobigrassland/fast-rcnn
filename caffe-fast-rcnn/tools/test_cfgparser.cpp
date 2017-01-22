/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "caffe/3rdparty/cfgparser.h"

#include <stdio.h>
#include <iostream>
#include <caffe/util/parse_config.hpp>

using std::string;
using std::vector;
using std::cout;
using std::endl;

/* main function */
int main(void)
{

    ParseConfig config("config.cfg");
    config.ParseTrainConfig();
    config.ParseTestConfig();
    config.ParseDeployConfig();
    config.ParseCommonConfig();
    /* load sample config */
    ConfigParser_t cfg;
    if (cfg.readFile("example.cfg"))
    {
        printf("Error: Cannot open config file 'example.cfg'\n");
        return 1;
    }

    string section, option;
    string testString;
    vector<string> testVector;
    int testInt;
    double testDouble;
    bool testBool;

    section = "section";
    option = "option1";

    /* get string value */
    if (cfg.getValue(section, option, &testString))
        cout << "Section/option [" << section << "] " << option
                << " found - value: '" << testString << "'" << endl;
    else
        cout << "Section/option [" << section << "] " << option
                << " not found" << endl;

    option = "option2";

    /* get int value */
    if (cfg.getValue(section, option, &testInt))
        cout << "Section/option [" << section << "] " << option
                << " found - value: " << testInt << endl;
    else
        cout << "Section/option [" << section << "] " << option
                << " not found" << endl;

    option = "option3";

    /* get double value */
    if (cfg.getValue(section, option, &testDouble))
        cout << "Section/option [" << section << "] " << option
                << " found - value: " << testDouble << endl;
    else
        cout << "Section/option [" << section << "] " << option
                << " not found" << endl;

    option = "option4";

    /* get bool value */
    if (cfg.getValue(section, option, &testBool)) {
        if (testBool)
            cout << "Section/option [" << section << "] " << option
                << " found - value: true" << endl;
        else
            cout << "Section/option [" << section << "] " << option
                << " found - value: false" << endl;
    }
    else
        cout << "Section/option [" << section << "] " << option
                << " not found" << endl;

    section = "multi";
    option = "multi";

    /* get vector<string> value */
    if (cfg.getValue(section, option, &testVector)) {
        vector<string>::const_iterator it = testVector.begin();
        cout << "Section/option [" << section << "] " << option
                << " - " << testVector.size() << " values: '" << *it << "'";
        for (it++; it != testVector.end(); it++)
            cout << ", '" << *it << "'";
        cout << endl;
    }
    else
        cout << "Section/option [" << section << "] " << option
                << " not found" << endl;

    section = "included";
    option = "included";

    /* get string value from included file */
    if (cfg.getValue(section, option, &testString))
        cout << "Included section/option [" << section << "] " << option
                << " found - value: '" << testString << "'" << endl;
    else
        cout << "Included section/option [" << section << "] " << option
                << " not found" << endl;


    section = "strings";
    option = "string1";

    /* get 1. string with special chars */
    if (cfg.getValue(section, option, &testString))
        cout << "Section/option [" << section << "] " << option
                << " found - value: '" << testString << "'" << endl;
    else
        cout << "Section/option [" << section << "] " << option
                << " not found" << endl;

    option = "string2";

    /* get 2. string with special chars */
    if (cfg.getValue(section, option, &testString))
        cout << "Section/option [" << section << "] " << option
                << " found - value: '" << testString << "'" << endl;
    else
        cout << "Section/option [" << section << "] " << option
                << " not found" << endl;

    option = "string3";

    /* get 3. string with special chars */
    if (cfg.getValue(section, option, &testString))
        cout << "Section/option [" << section << "] " << option
                << " found - value: '" << testString << "'" << endl;
    else
        cout << "Section/option [" << section << "] " << option
                << " not found" << endl;

    return 0;
}