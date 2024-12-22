/*分隔链表*/

#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
using namespace std;

struct ListNode{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

void printList(ListNode *head){
    while(head){
        cout << head->val << "->";
        head = head->next;
    }
    cout << "NULL" << endl;
}
class Solution {
    public:
    ListNode* partition(ListNode* head, int x){
        if (!head || !head->next) return head;

        ListNode *lh = new ListNode(0), *l = lh;
        ListNode *rh = new ListNode(0), *r = rh;

        ListNode *cur = head;
        while(cur){
            if (cur->val < x){
                l->next = new ListNode(cur->val);
                l = l->next;
            } else{
                r->next = new ListNode(cur->val);
                r = r->next;
            }
            cur = cur->next;
        }

        l->next = rh->next;
        r->next = nullptr;
        return lh->next;
    }
};

int main() {
    ListNode* l1 = new ListNode(1);
    l1->next = new ListNode(2);
    l1->next->next = new ListNode(2);
    l1->next->next->next = new ListNode(3);
    l1->next->next->next->next = new ListNode(3);
    l1->next->next->next->next->next = new ListNode(4);
    l1->next->next->next->next->next->next = new ListNode(6);

    ListNode* ans = partition(l1);
    printList(ans);
    return 0;
}


