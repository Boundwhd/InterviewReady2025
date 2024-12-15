#include <iostream>
#include <stdio.h>

#include <vector>
using namespace std;

void msort(vector<int> &nums, int left, int mid, int right){
    int n1 = mid - left + 1;    //左数组长度
    int n2 = right - mid;       //右数组长度

    vector<int> leftnums(n1), rightnums(n2);
    //创建两个数组，把两个区间放进去

    for(int i = 0;i < n1;i++){
        leftnums[i] = nums[left + i];
    }
    for(int i = 0;i < n2;i++){
        rightnums[i] = nums[mid + 1 + i];
    }

    //合并
    int i = 0, j = 0, k = left;     //索引初始化
    while(i < n1 && j < n2){
        if(leftnums[i] < rightnums[j]){
            nums[k] = leftnums[i];
            i++;
        }else{
            nums[k] = rightnums[j];
            j++;
        }
        k++;
    }
    //复制剩余的元素
    while(i < n1){
        nums[k] = leftnums[i];
        k++;
        i++;
    }

    while(j < n2){
        nums[k] = rightnums[j];
        k++;
        j++;
    }
}

void merge_sort(vector<int>& nums, int left, int right){
    if(left >= right) return;
        int mid = left + (right - left) / 2;
        merge_sort(nums, left, mid);
        merge_sort(nums, mid + 1, right);
        msort(nums, left, mid, right);
    
};

void merge_sort(vector<int>& nums){
    if (nums.size() <= 1) return;
    merge_sort(nums, 0, nums.size() - 1);
}

void printNums(vector<int>& nums){
    for(int i = 0;i < nums.size();i++){
        cout << nums[i] << " ";
    }
    cout << endl;
}

int main(){
    vector<int> nums = {3, 7, 6, 8, 2, 5, 9, 1, 0};
    cout << "Original input:" << " ";
    printNums(nums);
    merge_sort(nums);
    cout << "Sorted output:" << " ";
    printNums(nums);
    return 0;
}