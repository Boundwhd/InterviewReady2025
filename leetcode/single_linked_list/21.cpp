/*合并两个有序链表*/

#include <stdio.h>
#include <iostream>

using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* mergeTwoLists(ListNode* list1, ListNode* list2){
    ListNode* dummy = new ListNode(0);
    ListNode* cur = dummy;
    while(list1 && list2){
        if(list1->val <= list2->val){
            cur->next = new ListNode(list1->val);
            list1 = list1->next;
        }else{
            cur->next = new ListNode(list2->val);
            list2 = list2->next;
        }
        cur = cur->next;
    }
    if (list1){
        cur->next = list1;
    }
    if (list2){
        cur->next = list2;
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

int main(){
    ListNode* l1 = new ListNode(0);
    l1->next = new ListNode(3);
    l1->next->next = new ListNode(8);

    ListNode* l2 = new ListNode(2);
    l2->next = new ListNode(4);
    l2->next->next = new ListNode(7);

    ListNode* ans = mergeTwoLists(l1, l2);
    printList(ans);
    return 0;
}