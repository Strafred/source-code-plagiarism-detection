import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        int number = sc.nextInt();
        reverse(number);
    }
    public static void reverse(int number)
    {
        int tempDigit = 0;
        while (number > 0){
            tempDigit = number % 10;
            System.out.print(tempDigit);
            number = number / 10;
        }
    }
}
