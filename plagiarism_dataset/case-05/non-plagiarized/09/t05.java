import java.util.Scanner;
public class t05 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        int num = sc.nextInt();
        reverse(num);
    }
    public static void reverse(int number) {
        String tmpNumber = String.valueOf(number);
        for (int i = tmpNumber.length() - 1; i > -1; i--) {
            System.out.print(tmpNumber.charAt(i));
        }
        System.out.println("");
    }
}
