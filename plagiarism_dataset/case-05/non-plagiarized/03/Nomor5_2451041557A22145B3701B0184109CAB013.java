import java.util.Scanner;
public class Nomor5_2451041557A22145B3701B0184109CAB013 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an integer : ");
        int number = sc.nextInt();
        reverse(number);
    }
    public static void reverse(int number){
        String strNumber = Integer.toString(number);
        String temp = "";
        for(int i = strNumber.length(); i>0;i--){
            temp +=i;
        }
        int tempNumber = Integer.parseInt(temp);
        System.out.println((tempNumber));
    }
}
