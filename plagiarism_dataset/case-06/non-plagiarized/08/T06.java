import java.util.Scanner;
public class T06 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int a[] = new int[10];
        for (int i = 0; i < 10; i++) {
            System.out.print("Read a number");
            a[i] = sc.nextInt();
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(a[9-i]);
        }
    }
}
