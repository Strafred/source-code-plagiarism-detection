public class Kasus5 {
    public static void reverse(int number) {
        if(number>0) {
            System.out.print(number%10);
            reverse(number/=10);
        }
    }
    public static void main(String[] args) {
        System.out.print("Enter an integer: ");
        java.util.Scanner input = new java.util.Scanner(System.in);
        int number = input.nextInt();
        reverse(number);
        System.out.println("");
    }
}
