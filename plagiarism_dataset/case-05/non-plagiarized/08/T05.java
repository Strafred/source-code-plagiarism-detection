import java.util.Scanner;
public class T05 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        int a = sc.nextInt();
        String b = String.valueOf(a);
        for (int i = 0; i < b.length(); i++) {
            System.out.print(b.charAt(b.length()-1-i));
        }
    }
}
