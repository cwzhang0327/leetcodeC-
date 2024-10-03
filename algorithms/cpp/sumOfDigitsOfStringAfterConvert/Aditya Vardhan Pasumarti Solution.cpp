#include <string>
#include <iostream>

using namespace std;

class Solution {
public:
    int getLucky(string s, int k) 
    {
        string result = "";

        for(char i : s)
        {
            result += to_string(int(i) - int('a') + 1);
        }

        for(int i = 0; i < k; i++)
        {
            int temp = 0;

            for(char k : result)
            {
                temp += k - '0';
            }

            result = to_string(temp);
        }

        return stoi(result);
    }
};

int main() {

    Solution sol;

    cout << sol.getLucky("iiii", 1) << endl;
    cout << sol.getLucky("leetcode", 2) << endl;
    cout << sol.getLucky("zbax", 2) << endl;

    return 0;
}