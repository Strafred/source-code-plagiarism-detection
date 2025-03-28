import java.util.Scanner;
public class T05 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        int inp = sc.nextInt();
        reverse(inp);
    }
    public static void reverse(int number){
        int a = 0;
        int i = (int)(Math.log10(number));
        while(number!=0){
            int b = number%10;
            a += b * Math.pow(10, i--);
            number/=10;
        }
        System.out.println(a);
    }
}
