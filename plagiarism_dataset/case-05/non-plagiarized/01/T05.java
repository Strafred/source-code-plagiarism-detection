import java.util.Scanner;
public class T05 {
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        int number = sc.nextInt();
        reverse(number);
    }
    public static void reverse(int number){
        while(number > 0){
            System.out.print(number % 10);
            number /= 10;
        }
        System.out.println();
    }
}
