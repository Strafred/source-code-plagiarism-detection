import java.util.Scanner;
public class Soal5 {
    public static void reverse(int number){
        String a = String.valueOf(number);
        for(int i = a.length()-1; i>-1; i--){
            System.out.print(a.charAt(i));
        }
        System.out.println("");
    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        int number = sc.nextInt();
        reverse(number);
    }
}
