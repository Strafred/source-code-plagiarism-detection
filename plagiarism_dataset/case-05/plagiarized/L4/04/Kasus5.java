public class Kasus5 {
    public static void reverse(int number) {
        System.out.print(number%10);
    }
    public static void main(String[] args) {
        System.out.print("Enter an integer: ");
        java.util.Scanner input = new java.util.Scanner(System.in);
        int number = input.nextInt();
        for( ; number>0 ; number/=10)
            reverse(number);
        System.out.println("");
    }
}
