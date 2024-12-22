/*旋转链表*/

#include <iostream>
#include <stdio.h>

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

ListNode* rotateRight(ListNode* head, int k){
    if (!head) return nullptr;

    ListNode* cur = head;
    int n = 1;
    while (cur->next) cur = cur->next, n++;
    cur->next = head;

    for (int i = 0;i < n - k % n; i++){
    cur = cur->next;
    }
    head = cur->next;
    cur->next = nullptr;

    return head;
}

int main(){
    ListNode* l1 = new ListNode(1);
    l1->next = new ListNode(2);
    l1->next->next = new ListNode(3);
    l1->next->next->next = new ListNode(4);
    l1->next->next->next->next = new ListNode(5);
    l1->next->next->next->next->next = new ListNode(6);
    l1->next->next->next->next->next->next = new ListNode(7);

    ListNode* ans = rotateRight(l1, 4);
    printList(ans);
    return 0;
}