public class Main {
    public static int[] nums;
    public static void main(String[] args) {
        java.util.Scanner input = new java.util.Scanner(System.in);
        nums = new int[10];
        for (int i = 0; i < 10; i++) {
            System.out.print(
                    "Read a number: ");
            nums[i] = input.nextInt();
        }
        outputs();
    }
    public static void outputs(){
        int i = 9;
        while(i >= 0){
            System.out.println(nums[i--]);
        }
    }
}
