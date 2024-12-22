/*翻转链表II*/

#include <iostream>
#include <stdio.h>

using namespace std;

struct ListNode{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseBetween(ListNode* head, int left, int right) {
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    ListNode* pre = dummy;
    for(int i = 0;i < left - 1; i++){
        pre = pre->next;
    }
    ListNode* cur = pre->next;
    for(int i = 0; i < right - left; i++){
        ListNode *next = cur->next;
        cur->next = next->next;
        next->next = pre->next;
        pre->next = next;
    }
    return dummy->next;
}

void printList(ListNode *head){
    while(head){
        cout << head->val << "->";
        head = head->next;
    }
    cout << "NULL" << endl;
}


int main() {
    ListNode* l1 = new ListNode(1);
    l1->next = new ListNode(2);
    l1->next->next = new ListNode(3);
    l1->next->next->next = new ListNode(4);
    l1->next->next->next->next = new ListNode(5);
    l1->next->next->next->next->next = new ListNode(6);
    l1->next->next->next->next->next->next = new ListNode(7);

    ListNode* ans = reverseBetween(l1, 2, 5);
    printList(ans);
    return 0;
}