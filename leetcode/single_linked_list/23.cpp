/*合并k个升序链表*/

#include <iostream>
#include <stdio.h>
#include <vector>
#include <queue>
using namespace std;

struct ListNode{
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};
//存放链表节点最小堆创建的比较函数
struct Cmp {
    bool operator() (ListNode* a, ListNode* b) {
        return a->val > b->val;
    }
};

ListNode* mergeKLists(vector<ListNode*>& lists) {
    priority_queue<ListNode*, vector<ListNode*>, Cmp> heap;
    for(int i = 0;i < lists.size();i++){
        if(lists[i]) heap.push(lists[i]);
    }

    ListNode* dummy = new ListNode(0), *cur = dummy;
    while(!heap.empty()){
        ListNode* top = heap.top();
        heap.pop();
        cur = cur->next = top;
        if (top->next) heap.push(top->next);
    }
    return dummy->next;
}

