import java.util.Scanner;
public class T05 {
    static int a;
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an integer:");
        int a = sc.nextInt();
        reverse(a);
    }
    public static void reverse(int number) {
        int reverse = 0;
        while (number != 0) {
            reverse = reverse * 10;
            reverse = reverse + number % 10;
            number = number / 10;
        }
        System.out.println(reverse);
    }
}
