#include <iostream>
#include <stdio.h>

#include <vector>
using namespace std;

void fast_sort(vector<int>& nums, int low, int high){
    if(low >= high) return;

    int basic = nums[low];
    int l = low, r = high;

    while(l < r){
        while(nums[r] > basic && l < r){
            r--;
        }
        if(l < r){
            nums[l] = nums[r];
            l++;
        }
        while(nums[l] < basic && l < r){
            l++;
        }
        if(l < r){
            nums[r] = nums[l];
            r--;
        }
    }
    nums[l] = basic;
    
    fast_sort(nums, low, l - 1);
    fast_sort(nums, l + 1, high);
}

void fast_sort(vector<int>& nums){
    if (nums.size() <= 1) return;
    fast_sort(nums, 0, nums.size());
}

void printNums(vector<int>& nums){
    for(int i = 0;i < nums.size();i++){
        cout << nums[i] << " ";
    }
    cout << endl;
}


int main(){
    vector<int> nums = {3, 7, 2, 1, 6, 3, 8};
    cout << "Original input:" << " ";
    printNums(nums);
    fast_sort(nums);
    cout << "Sorted output:" << " ";
    printNums(nums);
    return 0;
}