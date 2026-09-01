nums = [4,1,5,8,96,3,1,74,87,5]

def kuaipai(nums,l,r):
    if l<r:
        mid = support(nums,l,r)
        kuaipai(nums,l,mid-1)
        kuaipai(nums,mid+1,r)

def support(nums,l,r):
    p = nums[r]
    i = l-1
    for j in range(l,r):
        if nums[j] <= p:
            i += 1
            nums[i], nums[j] = nums[j], nums[i]
    nums[i+1], nums[r] = nums[r], nums[i+1]
    return  i+1

print(nums)
kuaipai(nums,0,len(nums)-1)
print(nums)
