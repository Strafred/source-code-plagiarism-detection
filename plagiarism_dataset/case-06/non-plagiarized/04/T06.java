import java.util.Scanner;
public class T06 {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        int[] a = new int[10];
        for (int i = 0; i < a.length; i++) {
            System.out.print("Read a number: ");
            a[i] = s.nextInt();
        }
        for (int i = a.length-1; i >= 0 ; i--) {
            System.out.println(a[i]);
        }
    }
}
